package config

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolvePathsRelative(t *testing.T) {
	tmp := t.TempDir()
	orig, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(tmp); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(orig) })

	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	paths, err := ResolvePaths("./.falkengo")
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	if paths.StateDir != filepath.Join(wd, ".falkengo") {
		t.Fatalf("StateDir = %q, want temp state dir", paths.StateDir)
	}
	if paths.ManifestPath != filepath.Join(wd, ".falkengo", "manifest.sqlite") {
		t.Fatalf("ManifestPath = %q", paths.ManifestPath)
	}
	if paths.VecgoPath != filepath.Join(wd, ".falkengo", "vecgo-data") {
		t.Fatalf("VecgoPath = %q", paths.VecgoPath)
	}
}

func TestResolvePathsAbsolute(t *testing.T) {
	state := filepath.Join(t.TempDir(), "state")
	paths, err := ResolvePaths(state)
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	if paths.StateDir != filepath.Clean(state) {
		t.Fatalf("StateDir = %q, want absolute state", paths.StateDir)
	}
}

func TestIsInsideStateDir(t *testing.T) {
	root := t.TempDir()
	paths, err := ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	tests := []struct {
		path string
		want bool
	}{
		{paths.StateDir, true},
		{filepath.Join(paths.StateDir, "manifest.sqlite"), true},
		{filepath.Join(paths.StateDir, "vecgo-data", "CURRENT"), true},
		{filepath.Join(root, "README.md"), false},
	}
	for _, tt := range tests {
		got, err := IsInsideStateDir(tt.path, paths)
		if err != nil {
			t.Fatalf("IsInsideStateDir(%q): %v", tt.path, err)
		}
		if got != tt.want {
			t.Fatalf("IsInsideStateDir(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

func TestAcquireWriteLockFailsWhenHeld(t *testing.T) {
	paths, err := ResolvePaths(filepath.Join(t.TempDir(), ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	lock, err := AcquireWriteLock(paths)
	if err != nil {
		t.Fatalf("AcquireWriteLock: %v", err)
	}
	defer lock.Release()

	second, err := AcquireWriteLock(paths)
	if err == nil {
		second.Release()
		t.Fatal("second lock succeeded, want error")
	}
}

func TestAcquireWriteLockRemovesDeadProcessLock(t *testing.T) {
	paths, err := ResolvePaths(filepath.Join(t.TempDir(), ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	if err := EnsureStateDirs(paths); err != nil {
		t.Fatalf("EnsureStateDirs: %v", err)
	}
	deadPID := exitedProcessPID(t)
	if processRunning(deadPID) {
		t.Skipf("platform cannot identify exited pid %d as stale", deadPID)
	}
	path := WriteLockPath(paths)
	if err := os.WriteFile(path, []byte(fmt.Sprintf("pid=%d\ncreated_at=2026-05-24T15:45:27Z\n", deadPID)), 0o600); err != nil {
		t.Fatalf("write stale lock: %v", err)
	}

	lock, err := AcquireWriteLock(paths)
	if err != nil {
		t.Fatalf("AcquireWriteLock: %v", err)
	}
	defer lock.Release()

	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if got := string(contents); got == "" || !containsPID(got, os.Getpid()) {
		t.Fatalf("lock contents = %q, want current process pid", got)
	}
}

func TestAcquireWriteLockHelperProcess(t *testing.T) {
	if os.Getenv("FALKEN_VECTOR_LOCK_HELPER") != "1" {
		return
	}
	os.Exit(0)
}

func exitedProcessPID(t *testing.T) int {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestAcquireWriteLockHelperProcess")
	cmd.Env = append(os.Environ(), "FALKEN_VECTOR_LOCK_HELPER=1")
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper process: %v", err)
	}
	pid := cmd.Process.Pid
	if err := cmd.Wait(); err != nil {
		t.Fatalf("wait helper process: %v", err)
	}
	return pid
}

func containsPID(contents string, pid int) bool {
	return strings.Contains(contents, fmt.Sprintf("pid=%d", pid))
}
