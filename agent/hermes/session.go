package hermes

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/chenhg5/cc-connect/core"
)

// hermesSession manages a multi-turn Hermes conversation.
// Each Send() spawns `hermes --output-format stream-json -p <prompt>`.
// Subsequent turns use `--resume <sessionID>` to continue the conversation.
type hermesSession struct {
	cmd       string
	workDir   string
	model     string
	mode      string
	extraEnv  []string
	events    chan core.Event
	sessionID atomic.Value // stores string
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	alive     atomic.Bool
}

func newHermesSession(ctx context.Context, cmd, workDir, model, mode, resumeID string, extraEnv []string) (*hermesSession, error) {
	sessionCtx, cancel := context.WithCancel(ctx)

	s := &hermesSession{
		cmd:      cmd,
		workDir:  workDir,
		model:    model,
		mode:     mode,
		extraEnv: extraEnv,
		events:   make(chan core.Event, 64),
		ctx:      sessionCtx,
		cancel:   cancel,
	}
	s.alive.Store(true)

	if resumeID != "" && resumeID != core.ContinueSession {
		s.sessionID.Store(resumeID)
	}

	return s, nil
}

func (s *hermesSession) Send(prompt string, images []core.ImageAttachment, files []core.FileAttachment) error {
	if len(images) > 0 {
		slog.Warn("hermesSession: images not supported, ignoring")
	}
	if len(files) > 0 {
		filePaths := core.SaveFilesToDisk(s.workDir, files)
		prompt = core.AppendFileRefs(prompt, filePaths)
	}
	if !s.alive.Load() {
		return fmt.Errorf("session is closed")
	}

	args := []string{"--output-format", "stream-json", "-p", prompt}

	sid := s.CurrentSessionID()
	if sid != "" {
		args = append(args, "--resume", sid)
	}

	if s.model != "" {
		args = append(args, "--model", s.model)
	}

	if s.mode == "yolo" {
		slog.Warn("hermesSession: launching in yolo mode — all permission checks bypassed")
		args = append(args, "--dangerously-skip-permissions")
	}

	slog.Debug("hermesSession: launching", "resume", sid != "", "args", core.RedactArgs(args))

	cmd := exec.CommandContext(s.ctx, s.cmd, args...)
	cmd.Dir = s.workDir
	env := os.Environ()
	if len(s.extraEnv) > 0 {
		env = core.MergeEnv(env, s.extraEnv)
	}
	cmd.Env = env

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("hermesSession: stdout pipe: %w", err)
	}

	var stderrBuf bytes.Buffer
	cmd.Stderr = &stderrBuf

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("hermesSession: start: %w", err)
	}

	s.wg.Add(1)
	go s.readLoop(cmd, stdout, &stderrBuf)

	return nil
}

func (s *hermesSession) readLoop(cmd *exec.Cmd, stdout io.ReadCloser, stderrBuf *bytes.Buffer) {
	defer s.wg.Done()
	defer func() {
		if err := cmd.Wait(); err != nil {
			stderrMsg := strings.TrimSpace(stderrBuf.String())
			if stderrMsg != "" {
				slog.Error("hermesSession: process failed", "error", err, "stderr", truncStr(stderrMsg, 200))
				evt := core.Event{Type: core.EventError, Error: fmt.Errorf("%s", stderrMsg)}
				select {
				case s.events <- evt:
				case <-s.ctx.Done():
					return
				}
			}
		}
	}()

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var raw map[string]any
		if err := json.Unmarshal([]byte(line), &raw); err != nil {
			slog.Debug("hermesSession: non-JSON line", "line", truncStr(line, 100))
			continue
		}

		s.handleEvent(raw)
	}

	if err := scanner.Err(); err != nil {
		slog.Error("hermesSession: scanner error", "error", err)
		evt := core.Event{Type: core.EventError, Error: fmt.Errorf("read stdout: %w", err)}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
	}

	// Emit EventResult when the process finishes.
	sid := s.CurrentSessionID()
	evt := core.Event{Type: core.EventResult, SessionID: sid, Done: true}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
	}
}

// handleEvent dispatches a parsed JSON event from the Hermes CLI.
// Hermes emits events in a format compatible with the Claude Code streaming
// protocol (stream-json), so we handle the same event types.
func (s *hermesSession) handleEvent(raw map[string]any) {
	eventType, _ := raw["type"].(string)

	switch eventType {
	case "system":
		// Session init event with session ID.
		if subtype, _ := raw["subtype"].(string); subtype == "init" {
			if id, ok := raw["session_id"].(string); ok && id != "" {
				s.sessionID.Store(id)
				slog.Debug("hermesSession: session started", "session_id", id)
			}
		}

	case "assistant":
		s.handleAssistantEvent(raw)

	case "result":
		content, _ := raw["result"].(string)
		sid := s.CurrentSessionID()
		evt := core.Event{Type: core.EventResult, Content: content, SessionID: sid, Done: true}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
		}

	default:
		slog.Debug("hermesSession: unhandled event", "type", eventType)
	}
}

// handleAssistantEvent processes assistant message events.
func (s *hermesSession) handleAssistantEvent(raw map[string]any) {
	message, _ := raw["message"].(map[string]any)
	if message == nil {
		return
	}

	content, _ := message["content"].([]any)
	for _, c := range content {
		item, _ := c.(map[string]any)
		if item == nil {
			continue
		}

		itemType, _ := item["type"].(string)
		switch itemType {
		case "text":
			text, _ := item["text"].(string)
			if text != "" {
				evt := core.Event{Type: core.EventText, Content: text}
				select {
				case s.events <- evt:
				case <-s.ctx.Done():
					return
				}
			}

		case "thinking":
			thinking, _ := item["thinking"].(string)
			if thinking != "" {
				evt := core.Event{Type: core.EventThinking, Content: thinking}
				select {
				case s.events <- evt:
				case <-s.ctx.Done():
					return
				}
			}

		case "tool_use":
			name, _ := item["name"].(string)
			input := extractToolInput(item)
			evt := core.Event{Type: core.EventToolUse, ToolName: name, ToolInput: input}
			select {
			case s.events <- evt:
			case <-s.ctx.Done():
				return
			}
		}
	}
}

// extractToolInput pulls a concise summary from a tool_use content item.
func extractToolInput(item map[string]any) string {
	inputRaw, _ := item["input"].(map[string]any)
	if inputRaw == nil {
		return ""
	}
	if cmd, ok := inputRaw["command"].(string); ok && cmd != "" {
		return cmd
	}
	if fp, ok := inputRaw["file_path"].(string); ok && fp != "" {
		return fp
	}
	if pattern, ok := inputRaw["pattern"].(string); ok && pattern != "" {
		return pattern
	}
	if query, ok := inputRaw["query"].(string); ok && query != "" {
		return query
	}
	if desc, ok := inputRaw["description"].(string); ok && desc != "" {
		return desc
	}
	b, _ := json.Marshal(inputRaw)
	return truncStr(string(b), 200)
}

func (s *hermesSession) RespondPermission(_ string, _ core.PermissionResult) error {
	// Hermes handles permissions internally: in "default" mode it prompts via
	// its own CLI UI, and in "yolo" mode it auto-approves all tool calls.
	// Interactive permission responses through cc-connect are not supported.
	return nil
}

func (s *hermesSession) Events() <-chan core.Event {
	return s.events
}

func (s *hermesSession) CurrentSessionID() string {
	v, _ := s.sessionID.Load().(string)
	return v
}

func (s *hermesSession) Alive() bool {
	return s.alive.Load()
}

func (s *hermesSession) Close() error {
	s.alive.Store(false)
	s.cancel()
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(8 * time.Second):
		slog.Warn("hermesSession: close timed out, abandoning wg.Wait")
	}
	close(s.events)
	return nil
}

func truncStr(s string, maxRunes int) string {
	if utf8.RuneCountInString(s) <= maxRunes {
		return s
	}
	return string([]rune(s)[:maxRunes]) + "..."
}
