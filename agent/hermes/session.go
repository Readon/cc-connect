package hermes

import (
	"bytes"
	"context"
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
// Each Send() spawns `hermes chat -q <prompt> -Q`.
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

	// Hermes uses `hermes chat -q <prompt> -Q` for non-interactive single-query
	// mode. -Q (--quiet) suppresses the banner and spinner, and outputs only the
	// final response text followed by a `session_id: <id>` line.
	args := []string{"chat", "-q", prompt, "-Q"}

	sid := s.CurrentSessionID()
	if sid != "" {
		args = append(args, "--resume", sid)
	}

	if s.model != "" {
		args = append(args, "--model", s.model)
	}

	if s.mode == "yolo" {
		slog.Warn("hermesSession: launching in yolo mode — all permission checks bypassed")
		args = append(args, "--yolo")
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

	// Buffer all output — hermes quiet mode emits the complete response at once.
	outputBytes, readErr := io.ReadAll(stdout)

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
		return
	}

	if readErr != nil {
		slog.Error("hermesSession: read stdout", "error", readErr)
		evt := core.Event{Type: core.EventError, Error: fmt.Errorf("read stdout: %w", readErr)}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
		}
		return
	}

	response, sessionID := parseHermesOutput(string(outputBytes))

	if sessionID != "" {
		s.sessionID.Store(sessionID)
		slog.Debug("hermesSession: session started", "session_id", sessionID)
	}

	if response != "" {
		evt := core.Event{Type: core.EventText, Content: response}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
			return
		}
	}

	sid := s.CurrentSessionID()
	evt := core.Event{Type: core.EventResult, SessionID: sid, Done: true}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
	}
}

// parseHermesOutput splits the quiet-mode output into the response text and the
// session ID.  Hermes prints the response followed by a blank line and then
// "session_id: <id>" (with a leading newline before the label).
func parseHermesOutput(output string) (response, sessionID string) {
	lines := strings.Split(output, "\n")

	// Scan from the end for the "session_id: " line.
	sidIdx := -1
	for i := len(lines) - 1; i >= 0; i-- {
		line := strings.TrimSpace(lines[i])
		if strings.HasPrefix(line, "session_id: ") {
			sessionID = strings.TrimPrefix(line, "session_id: ")
			sidIdx = i
			break
		}
	}

	// Everything before the session_id line (minus trailing blank lines) is the response.
	if sidIdx >= 0 {
		response = strings.TrimRight(strings.Join(lines[:sidIdx], "\n"), "\n")
	} else {
		response = strings.TrimRight(output, "\n")
	}

	return
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
