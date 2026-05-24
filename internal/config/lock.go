package config

import (
	"errors"
	"fmt"
	"os"
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
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return nil, fmt.Errorf("%w: %s", ErrLockHeld, path)
		}
		return nil, err
	}
	_, _ = fmt.Fprintf(file, "pid=%d\ncreated_at=%s\n", os.Getpid(), time.Now().UTC().Format(time.RFC3339Nano))
	if err := file.Close(); err != nil {
		_ = os.Remove(path)
		return nil, err
	}
	return &WriteLock{path: path}, nil
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
