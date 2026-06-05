package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadLockInfo(t *testing.T) {
	t.Run("Valid pid", func(t *testing.T) {
		tempDir := t.TempDir()
		lockPath := filepath.Join(tempDir, "lock1.txt")
		err := os.WriteFile(lockPath, []byte("12345"), 0644)
		if err != nil {
			t.Fatalf("Failed to write mock file: %v", err)
		}

		pid, err := loadLockInfo(lockPath)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if pid != 12345 {
			t.Errorf("Expected PID 12345, got %d", pid)
		}
	})

	t.Run("Valid pid with whitespace", func(t *testing.T) {
		tempDir := t.TempDir()
		lockPath := filepath.Join(tempDir, "lock2.txt")
		err := os.WriteFile(lockPath, []byte("  98765 \n"), 0644)
		if err != nil {
			t.Fatalf("Failed to write mock file: %v", err)
		}

		pid, err := loadLockInfo(lockPath)
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}
		if pid != 98765 {
			t.Errorf("Expected PID 98765, got %d", pid)
		}
	})

	t.Run("File not found", func(t *testing.T) {
		tempDir := t.TempDir()
		lockPath := filepath.Join(tempDir, "nonexistent.txt")

		pid, err := loadLockInfo(lockPath)
		if err == nil {
			t.Error("Expected an error for non-existent file, got nil")
		}
		if pid != 0 {
			t.Errorf("Expected PID 0 on error, got %d", pid)
		}
	})

	t.Run("Invalid data", func(t *testing.T) {
		tempDir := t.TempDir()
		lockPath := filepath.Join(tempDir, "lock3.txt")
		err := os.WriteFile(lockPath, []byte("not_a_number"), 0644)
		if err != nil {
			t.Fatalf("Failed to write mock file: %v", err)
		}

		pid, err := loadLockInfo(lockPath)
		if err == nil {
			t.Error("Expected an error for invalid data, got nil")
		}
		if pid != 0 {
			t.Errorf("Expected PID 0 on error, got %d", pid)
		}
	})
}
