package config

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

var ErrLockHeld = errors.New("write lock is already held")

type WriteLock struct {
	path string
	file *os.File
}

func AcquireWriteLock(paths Paths) (*WriteLock, error) {
	if err := EnsureStateDirs(paths); err != nil {
		return nil, err
	}
	path := WriteLockPath(paths)
	lock, err := createWriteLock(path)
	if err == nil {
		return lock, nil
	}
	if !errors.Is(err, os.ErrExist) {
		return nil, err
	}
	removed, removeErr := removeStaleWriteLock(path)
	if removeErr != nil {
		return nil, removeErr
	}
	if removed {
		lock, err = createWriteLock(path)
		if err == nil {
			return lock, nil
		}
		if !errors.Is(err, os.ErrExist) {
			return nil, err
		}
	}
	return nil, writeLockHeldError(path)
}

func createWriteLock(path string) (*WriteLock, error) {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return nil, err
	}
	_, _ = fmt.Fprintf(file, "pid=%d\ncreated_at=%s\n", os.Getpid(), time.Now().UTC().Format(time.RFC3339Nano))
	if err := file.Close(); err != nil {
		_ = os.Remove(path)
		return nil, err
	}
	return &WriteLock{path: path}, nil
}

type writeLockInfo struct {
	pid int
}

func removeStaleWriteLock(path string) (bool, error) {
	info, err := readWriteLockInfo(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return true, nil
		}
		return false, nil
	}
	if info.pid <= 0 || processRunning(info.pid) {
		return false, nil
	}
	if err := os.Remove(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return true, nil
		}
		return false, fmt.Errorf("remove stale write lock %s: %w", path, err)
	}
	return true, nil
}

func writeLockHeldError(path string) error {
	info, err := readWriteLockInfo(path)
	if err == nil && info.pid > 0 {
		return fmt.Errorf("%w: %s (pid %d)", ErrLockHeld, path, info.pid)
	}
	return fmt.Errorf("%w: %s", ErrLockHeld, path)
}

func readWriteLockInfo(path string) (writeLockInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return writeLockInfo{}, err
	}
	defer file.Close()

	info := writeLockInfo{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		key, value, ok := strings.Cut(scanner.Text(), "=")
		if !ok || strings.TrimSpace(key) != "pid" {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(value))
		if err != nil {
			continue
		}
		info.pid = pid
	}
	if err := scanner.Err(); err != nil {
		return writeLockInfo{}, err
	}
	return info, nil
}

func (l *WriteLock) Release() error {
	if l == nil {
		return nil
	}
	var err error
	if l.file != nil {
		err = l.file.Close()
	}
	if removeErr := os.Remove(l.path); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) && err == nil {
		err = removeErr
	}
	return err
}

func loadLockInfo(path string) (int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}

	pidStr := strings.TrimSpace(string(data))
	pid, err := strconv.Atoi(pidStr)
	if err != nil {
		return 0, err
	}

	return pid, nil
}
