package ingest

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestFindCandidateFilesSkipsAndSorts(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "b.md"), "b")
	writeFile(t, filepath.Join(root, "a.go"), "a")
	writeFile(t, filepath.Join(root, "nested", "c.txt"), "c")
	writeFile(t, filepath.Join(root, ".falkengo", "hidden.md"), "state")
	writeFile(t, filepath.Join(root, "node_modules", "pkg", "x.js"), "dep")
	writeFile(t, filepath.Join(root, "image.png"), "png")

	files, err := FindCandidateFiles(context.Background(), WalkOptions{
		Root:     root,
		StateDir: filepath.Join(root, ".falkengo"),
	})
	if err != nil {
		t.Fatalf("FindCandidateFiles: %v", err)
	}
	got := fileNames(root, files)
	want := []string{"a.go", "b.md", filepath.Join("nested", "c.txt")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("files = %v, want %v", got, want)
	}
}

func TestFindCandidateFilesFiltersExtensions(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "a.md"), "a")
	writeFile(t, filepath.Join(root, "b.go"), "b")
	files, err := FindCandidateFiles(context.Background(), WalkOptions{
		Root:       root,
		StateDir:   filepath.Join(root, ".falkengo"),
		Extensions: []string{".md"},
	})
	if err != nil {
		t.Fatalf("FindCandidateFiles: %v", err)
	}
	got := fileNames(root, files)
	if !reflect.DeepEqual(got, []string{"a.md"}) {
		t.Fatalf("files = %v, want only md", got)
	}
}

func TestFindCandidateFilesIncludesEnvExample(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".env.example"), "PK=example")
	files, err := FindCandidateFiles(context.Background(), WalkOptions{
		Root:     root,
		StateDir: filepath.Join(root, ".falkengo"),
	})
	if err != nil {
		t.Fatalf("FindCandidateFiles: %v", err)
	}
	got := fileNames(root, files)
	if !reflect.DeepEqual(got, []string{".env.example"}) {
		t.Fatalf("files = %v, want .env.example", got)
	}
}

func writeFile(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func fileNames(root string, files []CandidateFile) []string {
	got := make([]string, 0, len(files))
	for _, file := range files {
		rel, _ := filepath.Rel(root, file.Path)
		got = append(got, rel)
	}
	return got
}
