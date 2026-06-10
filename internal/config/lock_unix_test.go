//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package config

import (
	"os"
	"os/exec"
	"testing"
)

func TestProcessRunning(t *testing.T) {
	tests := []struct {
		name string
		pid  int
		want bool
	}{
		{"zero", 0, false},
		{"negative", -1, false},
		{"self", os.Getpid(), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := processRunning(tt.pid); got != tt.want {
				t.Errorf("processRunning(%d) = %v, want %v", tt.pid, got, tt.want)
			}
		})
	}

	t.Run("dead_process", func(t *testing.T) {
		cmd := exec.Command("true")
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		pid := cmd.Process.Pid
		if err := cmd.Wait(); err != nil {
			t.Fatal(err)
		}
		if got := processRunning(pid); got {
			t.Errorf("processRunning(%d) = %v, want false", pid, got)
		}
	})
}
