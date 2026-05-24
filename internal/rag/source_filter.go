package rag

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

type SourceFilter struct {
	IncludeGlobs []string `json:"include,omitempty"`
	ExcludeGlobs []string `json:"exclude,omitempty"`
	SourceRoots  []string `json:"source_roots,omitempty"`
	BaseDir      string   `json:"-"`
}

func ParseSourceFilter(include []string, exclude []string, roots []string, cwd string) (SourceFilter, error) {
	if cwd == "" {
		var err error
		cwd, err = filepath.Abs(".")
		if err != nil {
			return SourceFilter{}, fmt.Errorf("resolve current directory: %w", err)
		}
	}
	cwd, err := filepath.Abs(cwd)
	if err != nil {
		return SourceFilter{}, fmt.Errorf("resolve current directory: %w", err)
	}
	filter := SourceFilter{
		IncludeGlobs: compactFilterStrings(include),
		ExcludeGlobs: compactFilterStrings(exclude),
		BaseDir:      filepath.Clean(cwd),
	}
	for _, root := range compactFilterStrings(roots) {
		if !filepath.IsAbs(root) {
			root = filepath.Join(cwd, root)
		}
		abs, err := filepath.Abs(root)
		if err != nil {
			return SourceFilter{}, fmt.Errorf("resolve source root %q: %w", root, err)
		}
		filter.SourceRoots = append(filter.SourceRoots, filepath.Clean(abs))
	}
	return filter, nil
}

func (f SourceFilter) IsEmpty() bool {
	return len(f.IncludeGlobs) == 0 && len(f.ExcludeGlobs) == 0 && len(f.SourceRoots) == 0
}

func (f SourceFilter) Match(path string, sourceRoot string) bool {
	if f.IsEmpty() {
		return true
	}
	if len(f.IncludeGlobs) != 0 {
		matched := false
		for _, pattern := range f.IncludeGlobs {
			if MatchGlob(pattern, path) {
				matched = true
				break
			}
		}
		if !matched {
			return false
		}
	}
	for _, pattern := range f.ExcludeGlobs {
		if MatchGlob(pattern, path) {
			return false
		}
	}
	if len(f.SourceRoots) == 0 {
		return true
	}
	for _, root := range f.SourceRoots {
		if pathUnderRoot(root, path, f.BaseDir) || sourceRootMatches(root, sourceRoot) {
			return true
		}
	}
	return false
}

func MatchGlob(pattern string, path string) bool {
	pattern = normalizeSlashPath(pattern)
	path = normalizeSlashPath(path)
	if pattern == "" || path == "" {
		return false
	}
	matcher, err := globRegexp(pattern)
	if err != nil {
		return false
	}
	for _, candidate := range globCandidates(path) {
		if matcher.MatchString(candidate) {
			return true
		}
	}
	return false
}

func compactFilterStrings(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func normalizeSlashPath(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return filepath.ToSlash(filepath.Clean(value))
}

func globCandidates(path string) []string {
	path = strings.TrimPrefix(path, "./")
	candidates := []string{path}
	trimmed := strings.TrimPrefix(path, "/")
	if trimmed != path {
		candidates = append(candidates, trimmed)
	}
	parts := strings.Split(trimmed, "/")
	for i := 1; i < len(parts); i++ {
		candidates = append(candidates, strings.Join(parts[i:], "/"))
	}
	return candidates
}

func globRegexp(pattern string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		ch := pattern[i]
		switch ch {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				b.WriteString(".*")
				i++
			} else {
				b.WriteString("[^/]*")
			}
		case '?':
			b.WriteString("[^/]")
		default:
			b.WriteString(regexp.QuoteMeta(string(ch)))
		}
	}
	b.WriteString("$")
	return regexp.Compile(b.String())
}

func pathUnderRoot(root string, path string, baseDir string) bool {
	root = filepath.Clean(root)
	candidates := []string{path}
	if path != "" && !filepath.IsAbs(path) && baseDir != "" {
		candidates = append(candidates, filepath.Join(baseDir, path))
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if isUnderRoot(root, candidate) {
			return true
		}
	}
	return false
}

func sourceRootMatches(root string, sourceRoot string) bool {
	if sourceRoot == "" {
		return false
	}
	return isUnderRoot(filepath.Clean(root), filepath.Clean(sourceRoot))
}

func isUnderRoot(root string, path string) bool {
	root = filepath.Clean(root)
	path = filepath.Clean(path)
	rel, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return rel == "." || (rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)))
}
