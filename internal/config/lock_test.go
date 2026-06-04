package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestWriteLockRelease(t *testing.T) {
	tmpDir := t.TempDir()
	lockPath := filepath.Join(tmpDir, "test.lock")

	// Test 1: Release a nil lock
	var lock *WriteLock
	if err := lock.Release(); err != nil {
		t.Errorf("expected no error releasing a nil lock, got %v", err)
	}

	// Test 2: Release a lock with a valid file and path
	file, err := os.Create(lockPath)
	if err != nil {
		t.Fatalf("failed to create lock file: %v", err)
	}
	lock = &WriteLock{
		path: lockPath,
		file: file,
	}

	if err := lock.Release(); err != nil {
		t.Errorf("expected no error releasing valid lock, got %v", err)
	}

	// Verify file is closed and deleted
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Errorf("expected lock file to be deleted, but it still exists")
	}

	// Test 3: Release a lock where the file is already closed
	file, err = os.Create(lockPath)
	if err != nil {
		t.Fatalf("failed to create lock file: %v", err)
	}
	file.Close() // Close it manually

	lock = &WriteLock{
		path: lockPath,
		file: file,
	}

	// Release should try to close it again (which errors in os.File) but proceed to delete
	err = lock.Release()
	if err == nil {
		t.Errorf("expected an error when releasing a lock with an already closed file")
	}

	// File should still be deleted
	if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Errorf("expected lock file to be deleted even if close failed")
	}

    // Test 4: Release a lock where file is nil but path exists
    file, err = os.Create(lockPath)
	if err != nil {
		t.Fatalf("failed to create lock file: %v", err)
	}
    file.Close() // Close it to avoid leaks, we only need the file to exist on disk

    lock = &WriteLock{
		path: lockPath,
		file: nil,
	}
    if err := lock.Release(); err != nil {
		t.Errorf("expected no error releasing lock with nil file but existing path, got %v", err)
	}

    if _, err := os.Stat(lockPath); !os.IsNotExist(err) {
		t.Errorf("expected lock file to be deleted")
	}

}

func TestWriteLockReleaseRemoveError(t *testing.T) {
	tmpDir := t.TempDir()
	lockPath := filepath.Join(tmpDir, "test.lock")

	// Create the file
	file, err := os.Create(lockPath)
	if err != nil {
		t.Fatalf("failed to create lock file: %v", err)
	}

	lock := &WriteLock{
		path: lockPath,
		file: file,
	}

	// Make the directory read-only so os.Remove will fail
	err = os.Chmod(tmpDir, 0555)
	if err != nil {
		t.Fatalf("failed to chmod dir: %v", err)
	}

	// Ensure we restore permissions so TempDir cleanup doesn't fail
	defer os.Chmod(tmpDir, 0755)

	err = lock.Release()
	if err == nil {
		t.Errorf("expected an error when releasing a lock and os.Remove fails")
	}
}
