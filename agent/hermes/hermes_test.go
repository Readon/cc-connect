package hermes

import (
	"testing"

	"github.com/chenhg5/cc-connect/core"
)

func TestNormalizeMode(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"yolo", "yolo"},
		{"YOLO", "yolo"},
		{"bypass", "yolo"},
		{"auto-approve", "yolo"},
		{"default", "default"},
		{"", "default"},
		{"unknown", "default"},
		{"  yolo  ", "yolo"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := normalizeMode(tt.input)
			if got != tt.expected {
				t.Errorf("normalizeMode(%q) = %q, want %q", tt.input, got, tt.expected)
			}
		})
	}
}

func TestAgent_Name(t *testing.T) {
	a := &Agent{}
	if got := a.Name(); got != "hermes" {
		t.Errorf("Name() = %q, want %q", got, "hermes")
	}
}

func TestAgent_CLIBinaryName(t *testing.T) {
	a := &Agent{}
	if got := a.CLIBinaryName(); got != "hermes" {
		t.Errorf("CLIBinaryName() = %q, want %q", got, "hermes")
	}
}

func TestAgent_CLIDisplayName(t *testing.T) {
	a := &Agent{}
	if got := a.CLIDisplayName(); got != "Hermes" {
		t.Errorf("CLIDisplayName() = %q, want %q", got, "Hermes")
	}
}

func TestAgent_SetWorkDir(t *testing.T) {
	a := &Agent{}
	a.SetWorkDir("/tmp/test")
	if got := a.GetWorkDir(); got != "/tmp/test" {
		t.Errorf("GetWorkDir() = %q, want %q", got, "/tmp/test")
	}
}

func TestAgent_SetModel(t *testing.T) {
	a := &Agent{}
	a.SetModel("claude-3-5-sonnet-20241022")
	a.mu.Lock()
	got := a.model
	a.mu.Unlock()
	if got != "claude-3-5-sonnet-20241022" {
		t.Errorf("model = %q, want %q", got, "claude-3-5-sonnet-20241022")
	}
}

func TestAgent_GetModel(t *testing.T) {
	a := &Agent{model: "claude-3-opus"}
	if got := a.GetModel(); got != "claude-3-opus" {
		t.Errorf("GetModel() = %q, want %q", got, "claude-3-opus")
	}
}

func TestAgent_SetMode(t *testing.T) {
	a := &Agent{}
	a.SetMode("yolo")
	if got := a.GetMode(); got != "yolo" {
		t.Errorf("GetMode() = %q, want %q", got, "yolo")
	}

	a.SetMode("default")
	if got := a.GetMode(); got != "default" {
		t.Errorf("GetMode() = %q, want %q", got, "default")
	}
}

func TestAgent_PermissionModes(t *testing.T) {
	a := &Agent{}
	modes := a.PermissionModes()
	if len(modes) != 2 {
		t.Errorf("PermissionModes() returned %d modes, want 2", len(modes))
	}
	if modes[0].Key != "default" {
		t.Errorf("modes[0].Key = %q, want %q", modes[0].Key, "default")
	}
	if modes[1].Key != "yolo" {
		t.Errorf("modes[1].Key = %q, want %q", modes[1].Key, "yolo")
	}
}

func TestNew_MissingBinary(t *testing.T) {
	_, err := New(map[string]any{
		"cmd": "hermes-binary-that-does-not-exist-xyz",
	})
	if err == nil {
		t.Error("expected error when binary not found in PATH")
	}
}

// verify Agent implements core.Agent
var _ core.Agent = (*Agent)(nil)

// ── parseHermesOutput tests ──────────────────────────────────────────────────

func TestParseHermesOutput_WithSessionID(t *testing.T) {
	output := "Hello, world!\n\nsession_id: 20260225_143052_a1b2c3\n"
	response, sessionID := parseHermesOutput(output)
	if response != "Hello, world!" {
		t.Errorf("response = %q, want %q", response, "Hello, world!")
	}
	if sessionID != "20260225_143052_a1b2c3" {
		t.Errorf("sessionID = %q, want %q", sessionID, "20260225_143052_a1b2c3")
	}
}

func TestParseHermesOutput_MultiLineResponse(t *testing.T) {
	output := "Line one\nLine two\nLine three\n\nsession_id: abc123\n"
	response, sessionID := parseHermesOutput(output)
	if response != "Line one\nLine two\nLine three" {
		t.Errorf("response = %q, want %q", response, "Line one\nLine two\nLine three")
	}
	if sessionID != "abc123" {
		t.Errorf("sessionID = %q, want %q", sessionID, "abc123")
	}
}

func TestParseHermesOutput_NoSessionID(t *testing.T) {
	output := "Just a response with no session id.\n"
	response, sessionID := parseHermesOutput(output)
	if response != "Just a response with no session id." {
		t.Errorf("response = %q, want %q", response, "Just a response with no session id.")
	}
	if sessionID != "" {
		t.Errorf("sessionID = %q, want empty", sessionID)
	}
}

func TestParseHermesOutput_EmptyOutput(t *testing.T) {
	response, sessionID := parseHermesOutput("")
	if response != "" {
		t.Errorf("response = %q, want empty", response)
	}
	if sessionID != "" {
		t.Errorf("sessionID = %q, want empty", sessionID)
	}
}
