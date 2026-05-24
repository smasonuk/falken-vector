package ingest

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/smasonuk/falken-vector/internal/config"
)

var DefaultExtensions = []string{".txt", ".md", ".go", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml"}

type WalkOptions struct {
	Root       string
	StateDir   string
	Extensions []string
}

type CandidateFile struct {
	Path       string
	SizeBytes  int64
	ModifiedAt time.Time
}

func FindCandidateFiles(ctx context.Context, opts WalkOptions) ([]CandidateFile, error) {
	root := strings.TrimSpace(opts.Root)
	if root == "" {
		root = "."
	}
	if !filepath.IsAbs(root) {
		wd, err := os.Getwd()
		if err != nil {
			return nil, err
		}
		root = filepath.Join(wd, root)
	}
	root = filepath.Clean(root)

	paths, err := config.ResolvePaths(opts.StateDir)
	if err != nil {
		return nil, err
	}
	extensions := extensionSet(opts.Extensions)
	files := make([]CandidateFile, 0)

	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if path == root {
			return nil
		}
		insideState, err := config.IsInsideStateDir(path, paths)
		if err != nil {
			return err
		}
		if insideState {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		name := entry.Name()
		if entry.IsDir() {
			if shouldSkipDir(name) || isHidden(name) {
				return filepath.SkipDir
			}
			return nil
		}
		hiddenAllowed := isHidden(name) && allowedHiddenTextFile(name, extensions)
		if isHidden(name) && !hiddenAllowed {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(name))
		if !hiddenAllowed && !extensions[ext] {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		files = append(files, CandidateFile{
			Path:       filepath.Clean(path),
			SizeBytes:  info.Size(),
			ModifiedAt: info.ModTime().UTC(),
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(files, func(i, j int) bool { return files[i].Path < files[j].Path })
	return files, nil
}

func extensionSet(values []string) map[string]bool {
	if len(values) == 0 {
		values = DefaultExtensions
	}
	set := make(map[string]bool, len(values))
	for _, value := range values {
		ext := strings.ToLower(strings.TrimSpace(value))
		if ext == "" {
			continue
		}
		if !strings.HasPrefix(ext, ".") {
			ext = "." + ext
		}
		set[ext] = true
	}
	return set
}

func shouldSkipDir(name string) bool {
	switch name {
	case ".git", "node_modules", "vendor", "dist", "build", "target", ".falkengo":
		return true
	default:
		return false
	}
}

func isHidden(name string) bool {
	return strings.HasPrefix(name, ".")
}

func allowedHiddenTextFile(name string, extensions map[string]bool) bool {
	if name == ".env.example" {
		return true
	}
	return extensions[strings.ToLower(filepath.Ext(name))]
}
