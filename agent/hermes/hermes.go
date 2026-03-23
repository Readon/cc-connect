package hermes

import (
	"context"
	"fmt"
	"log/slog"
	"os/exec"
	"strings"
	"sync"

	"github.com/chenhg5/cc-connect/core"
)

func init() {
	core.RegisterAgent("hermes", New)
}

// Agent drives the Hermes AI agent CLI (`hermes`).
type Agent struct {
	cmd        string // path to hermes binary
	workDir    string
	model      string
	mode       string // "default" | "yolo"
	sessionEnv []string
	mu         sync.Mutex
}

func New(opts map[string]any) (core.Agent, error) {
	workDir, _ := opts["work_dir"].(string)
	if workDir == "" {
		workDir = "."
	}
	model, _ := opts["model"].(string)
	mode, _ := opts["mode"].(string)
	mode = normalizeMode(mode)

	cmd, _ := opts["cmd"].(string)
	if cmd == "" {
		cmd = "hermes"
	}

	if _, err := exec.LookPath(cmd); err != nil {
		return nil, fmt.Errorf("hermes: %q not found in PATH, see https://github.com/NousResearch/hermes-agent for installation", cmd)
	}

	return &Agent{
		cmd:     cmd,
		workDir: workDir,
		model:   model,
		mode:    mode,
	}, nil
}

func normalizeMode(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "yolo", "bypass", "auto-approve":
		return "yolo"
	default:
		return "default"
	}
}

func (a *Agent) Name() string           { return "hermes" }
func (a *Agent) CLIBinaryName() string  { return "hermes" }
func (a *Agent) CLIDisplayName() string { return "Hermes" }

func (a *Agent) SetWorkDir(dir string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.workDir = dir
	slog.Info("hermes: work_dir changed", "work_dir", dir)
}

func (a *Agent) GetWorkDir() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.workDir
}

func (a *Agent) SetModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = model
	slog.Info("hermes: model changed", "model", model)
}

func (a *Agent) GetModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.model
}

func (a *Agent) AvailableModels(_ context.Context) []core.ModelOption {
	// Hermes uses its own model registry; configure models via the Hermes
	// configuration files rather than through cc-connect's model selection.
	return nil
}

func (a *Agent) SetSessionEnv(env []string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.sessionEnv = env
}

func (a *Agent) StartSession(ctx context.Context, sessionID string) (core.AgentSession, error) {
	a.mu.Lock()
	mode := a.mode
	model := a.model
	extraEnv := append([]string{}, a.sessionEnv...)
	a.mu.Unlock()
	return newHermesSession(ctx, a.cmd, a.workDir, model, mode, sessionID, extraEnv)
}

func (a *Agent) ListSessions(_ context.Context) ([]core.AgentSessionInfo, error) {
	return nil, nil
}

func (a *Agent) Stop() error { return nil }

// ── ModeSwitcher ─────────────────────────────────────────────

func (a *Agent) SetMode(mode string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.mode = normalizeMode(mode)
	slog.Info("hermes: mode changed", "mode", a.mode)
}

func (a *Agent) GetMode() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.mode
}

func (a *Agent) PermissionModes() []core.PermissionModeInfo {
	return []core.PermissionModeInfo{
		{Key: "default", Name: "Default", NameZh: "默认", Desc: "Standard permissions", DescZh: "标准权限模式"},
		{Key: "yolo", Name: "YOLO", NameZh: "全自动", Desc: "Auto-approve all tool calls", DescZh: "自动批准所有工具调用"},
	}
}
