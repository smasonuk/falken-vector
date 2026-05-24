package rag

import (
	"path/filepath"
	"testing"
)

func TestSourceFilterIncludeGlob(t *testing.T) {
	filter, err := ParseSourceFilter([]string{"internal/rag/**"}, nil, nil, "/repo")
	if err != nil {
		t.Fatal(err)
	}
	if !filter.Match("/repo/internal/rag/retrieve.go", "") {
		t.Fatal("include glob did not match expected path")
	}
	if filter.Match("/repo/internal/cli/query.go", "") {
		t.Fatal("include glob matched unexpected path")
	}
}

func TestSourceFilterExcludeGlob(t *testing.T) {
	filter, err := ParseSourceFilter(nil, []string{"internal/vendor/**"}, nil, "/repo")
	if err != nil {
		t.Fatal(err)
	}
	if filter.Match("/repo/internal/vendor/lib.go", "") {
		t.Fatal("exclude glob matched as included")
	}
	if !filter.Match("/repo/internal/rag/retrieve.go", "") {
		t.Fatal("exclude glob filtered unrelated path")
	}
}

func TestSourceFilterExcludeWinsOverInclude(t *testing.T) {
	filter, err := ParseSourceFilter([]string{"internal/**"}, []string{"internal/vendor/**"}, nil, "/repo")
	if err != nil {
		t.Fatal(err)
	}
	if filter.Match("/repo/internal/vendor/lib.go", "") {
		t.Fatal("exclude did not win over include")
	}
}

func TestSourceFilterSourceRootMatchesDocumentPath(t *testing.T) {
	root := filepath.Join(t.TempDir(), "docs")
	filter, err := ParseSourceFilter(nil, nil, []string{root}, "/repo")
	if err != nil {
		t.Fatal(err)
	}
	if !filter.Match(filepath.Join(root, "guide.md"), "") {
		t.Fatal("source root did not match document path")
	}
	if filter.Match(filepath.Join(filepath.Dir(root), "src", "main.go"), "") {
		t.Fatal("source root matched outside path")
	}
}

func TestSourceFilterSourceRootMatchesDocumentSourceRoot(t *testing.T) {
	root := filepath.Join(t.TempDir(), "docs")
	filter, err := ParseSourceFilter(nil, nil, []string{root}, "/repo")
	if err != nil {
		t.Fatal(err)
	}
	if !filter.Match("guide.md", root) {
		t.Fatal("source root did not match document source root")
	}
	if !filter.Match("guide.md", filepath.Join(root, "nested")) {
		t.Fatal("source root did not match nested document source root")
	}
}

func TestMatchGlobSupportsDoubleStarAndBasename(t *testing.T) {
	if !MatchGlob("docs/**", "/repo/docs/guide/setup.md") {
		t.Fatal("docs/** did not match nested docs path")
	}
	if !MatchGlob("*.md", "/repo/docs/guide/setup.md") {
		t.Fatal("*.md did not match basename")
	}
	if MatchGlob("docs/*.md", "/repo/docs/guide/setup.md") {
		t.Fatal("single star matched across path segment")
	}
}
