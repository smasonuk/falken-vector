package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResetCommandPromptsAndDeletes(t *testing.T) {
	tempDir := t.TempDir()
	stateDir := filepath.Join(tempDir, ".falkengo")
	if err := os.Mkdir(stateDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	var in bytes.Buffer
	in.WriteString("y\n") // Send 'y' to prompt

	cmd.SetOut(&out)
	cmd.SetIn(&in)
	cmd.SetArgs([]string{"--state-dir", stateDir, "reset"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	output := out.String()
	if !strings.Contains(output, "[y/N]") {
		t.Fatalf("output missing prompt: %q", output)
	}
	if !strings.Contains(output, "deleted "+stateDir) {
		t.Fatalf("output missing success message: %q", output)
	}

	if _, err := os.Stat(stateDir); !os.IsNotExist(err) {
		t.Fatalf("state dir was not deleted: %v", err)
	}
}

func TestResetCommandYesFlagSkipsPromptAndDeletes(t *testing.T) {
	tempDir := t.TempDir()
	stateDir := filepath.Join(tempDir, ".falkengo")
	if err := os.Mkdir(stateDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	var in bytes.Buffer
	// No input sent; should not block because of --yes flag

	cmd.SetOut(&out)
	cmd.SetIn(&in)
	cmd.SetArgs([]string{"--state-dir", stateDir, "reset", "--yes"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	output := out.String()
	if strings.Contains(output, "[y/N]") {
		t.Fatalf("output should not contain prompt: %q", output)
	}
	if !strings.Contains(output, "deleted "+stateDir) {
		t.Fatalf("output missing success message: %q", output)
	}

	if _, err := os.Stat(stateDir); !os.IsNotExist(err) {
		t.Fatalf("state dir was not deleted: %v", err)
	}
}

func TestResetCommandCancelsOnNo(t *testing.T) {
	tempDir := t.TempDir()
	stateDir := filepath.Join(tempDir, ".falkengo")
	if err := os.Mkdir(stateDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	var in bytes.Buffer
	in.WriteString("n\n") // Send 'n' to prompt

	cmd.SetOut(&out)
	cmd.SetIn(&in)
	cmd.SetArgs([]string{"--state-dir", stateDir, "reset"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	output := out.String()
	if !strings.Contains(output, "[y/N]") {
		t.Fatalf("output missing prompt: %q", output)
	}
	if !strings.Contains(output, "reset cancelled") {
		t.Fatalf("output missing cancellation message: %q", output)
	}

	if _, err := os.Stat(stateDir); err != nil {
		t.Fatalf("state dir should still exist: %v", err)
	}
}

func TestResetCommandFailsValidationWithoutForce(t *testing.T) {
	tempDir := t.TempDir()
	stateDir := filepath.Join(tempDir, "custom-dir") // does not end in .falkengo
	if err := os.Mkdir(stateDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	var in bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetIn(&in)
	cmd.SetArgs([]string{"--state-dir", stateDir, "reset"})

	err := cmd.Execute()
	if err == nil {
		t.Fatalf("expected validation error but got nil")
	}
	if !strings.Contains(err.Error(), "basename is not .falkengo") {
		t.Fatalf("unexpected error message: %v", err)
	}

	if _, err := os.Stat(stateDir); err != nil {
		t.Fatalf("state dir should still exist: %v", err)
	}
}

func TestResetCommandForceFlagOverridesValidation(t *testing.T) {
	tempDir := t.TempDir()
	stateDir := filepath.Join(tempDir, "custom-dir") // does not end in .falkengo
	if err := os.Mkdir(stateDir, 0755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	var in bytes.Buffer
	in.WriteString("y\n")

	cmd.SetOut(&out)
	cmd.SetIn(&in)
	cmd.SetArgs([]string{"--state-dir", stateDir, "reset", "--force-state-dir"})

	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}

	output := out.String()
	if !strings.Contains(output, "deleted "+stateDir) {
		t.Fatalf("output missing success message: %q", output)
	}

	if _, err := os.Stat(stateDir); !os.IsNotExist(err) {
		t.Fatalf("state dir was not deleted: %v", err)
	}
}
