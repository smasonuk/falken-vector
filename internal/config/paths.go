package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	DefaultStateDir   = "./.falkengo"
	ManifestFileName  = "manifest.sqlite"
	VecgoDirName      = "vecgo-data"
	LocksDirName      = "locks"
	WriteLockFileName = "write.lock"
)

type Paths struct {
	StateDir     string
	ManifestPath string
	VecgoPath    string
	LocksPath    string
}

func ResolvePaths(stateDir string) (Paths, error) {
	if strings.TrimSpace(stateDir) == "" {
		stateDir = DefaultStateDir
	}
	if !filepath.IsAbs(stateDir) {
		wd, err := os.Getwd()
		if err != nil {
			return Paths{}, err
		}
		stateDir = filepath.Join(wd, stateDir)
	}

	stateDir = filepath.Clean(stateDir)
	return Paths{
		StateDir:     stateDir,
		ManifestPath: filepath.Clean(filepath.Join(stateDir, ManifestFileName)),
		VecgoPath:    filepath.Clean(filepath.Join(stateDir, VecgoDirName)),
		LocksPath:    filepath.Clean(filepath.Join(stateDir, LocksDirName)),
	}, nil
}

func EnsureStateDirs(paths Paths) error {
	if err := os.MkdirAll(paths.StateDir, 0o755); err != nil {
		return err
	}
	return os.MkdirAll(paths.LocksPath, 0o755)
}

func IsInsideStateDir(path string, paths Paths) (bool, error) {
	if strings.TrimSpace(path) == "" {
		return false, nil
	}
	if !filepath.IsAbs(path) {
		wd, err := os.Getwd()
		if err != nil {
			return false, err
		}
		path = filepath.Join(wd, path)
	}
	target := filepath.Clean(path)
	state := filepath.Clean(paths.StateDir)
	if target == state {
		return true, nil
	}
	rel, err := filepath.Rel(state, target)
	if err != nil {
		return false, err
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)), nil
}

func WriteLockPath(paths Paths) string {
	return filepath.Join(paths.LocksPath, WriteLockFileName)
}

func ValidateResetStateDir(paths Paths, force bool) error {
	state := filepath.Clean(paths.StateDir)
	if state == "" {
		return fmt.Errorf("refusing to delete empty state directory")
	}
	if !filepath.IsAbs(state) {
		return fmt.Errorf("refusing to delete non-absolute state directory %q", state)
	}
	if filepath.Dir(state) == state {
		return fmt.Errorf("refusing to delete filesystem root %q", state)
	}
	wd, err := os.Getwd()
	if err != nil {
		return err
	}
	if samePath(state, wd) {
		return fmt.Errorf("refusing to delete current working directory %q", state)
	}
	if home, err := os.UserHomeDir(); err == nil && home != "" && samePath(state, home) {
		return fmt.Errorf("refusing to delete user home directory %q", state)
	}
	if filepath.Base(state) != ".falkengo" && !force {
		return fmt.Errorf("refusing to delete state directory %q because its basename is not .falkengo; pass --force-state-dir to override", state)
	}
	return nil
}

func samePath(a, b string) bool {
	rel, err := filepath.Rel(filepath.Clean(a), filepath.Clean(b))
	return err == nil && rel == "."
}
