package ingest

import (
	"context"
	"errors"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/smasonuk/falken-vector/internal/config"
)

var DefaultExtensions = []string{".txt", ".md", ".go", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml", ".toml"}

const textSniffBytes = 8192

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
	restrictExtensions := len(extensions) != 0
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
		if !hiddenAllowed && restrictExtensions && !extensions[ext] {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		textLike, err := isTextLikeFile(path)
		if err != nil {
			return err
		}
		if !textLike {
			return nil
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

func isTextLikeFile(path string) (bool, error) {
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()

	buf := make([]byte, textSniffBytes)
	n, err := file.Read(buf)
	if err != nil && !errors.Is(err, io.EOF) {
		return false, err
	}
	if n == 0 {
		return true, nil
	}
	sample := buf[:n]
	for _, b := range sample {
		if b == 0 {
			return false, nil
		}
	}
	if !validUTF8Sample(sample, n == textSniffBytes) {
		return false, nil
	}
	control := 0
	for _, b := range sample {
		if b < 0x20 && b != '\t' && b != '\n' && b != '\r' && b != '\f' {
			control++
		}
	}
	return control*10 <= len(sample)*3, nil
}

func validUTF8Sample(sample []byte, mayEndMidRune bool) bool {
	if utf8.Valid(sample) {
		return true
	}
	if !mayEndMidRune {
		return false
	}
	for trim := 1; trim < utf8.UTFMax && trim < len(sample); trim++ {
		if utf8.Valid(sample[:len(sample)-trim]) {
			return true
		}
	}
	return false
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
