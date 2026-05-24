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
	writeBytes(t, filepath.Join(root, "image.png"), []byte{0x89, 'P', 'N', 'G', 0x00, 0x01})

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

func TestFindCandidateFilesIncludesTextLikeFilesByContent(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "README"), "plain text")
	writeFile(t, filepath.Join(root, "Dockerfile"), "FROM scratch\n")
	writeFile(t, filepath.Join(root, "config.conf"), "key=value\n")
	writeFile(t, filepath.Join(root, "apis"), "GET /v1/things\n")
	writeBytes(t, filepath.Join(root, "binary"), []byte{0x00, 0x01, 0x02})
	writeBytes(t, filepath.Join(root, "invalid"), []byte{0xff, 0xfe, 0xfd})

	files, err := FindCandidateFiles(context.Background(), WalkOptions{
		Root:     root,
		StateDir: filepath.Join(root, ".falkengo"),
	})
	if err != nil {
		t.Fatalf("FindCandidateFiles: %v", err)
	}
	got := fileNames(root, files)
	want := []string{"Dockerfile", "README", "apis", "config.conf"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("files = %v, want text-like files %v", got, want)
	}
}

func TestFindCandidateFilesFiltersExtensions(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "a.md"), "a")
	writeFile(t, filepath.Join(root, "b.go"), "b")
	writeFile(t, filepath.Join(root, "README"), "readme")
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
	writeFile(t, filepath.Join(root, ".env.example"), "FALKENGO_EMBEDDING_MODEL_API_KEY=example")
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

func TestFindCandidateFilesAllowsHiddenFileWithExplicitExtension(t *testing.T) {
	root := t.TempDir()
	writeFile(t, filepath.Join(root, ".hidden.md"), "hidden")
	files, err := FindCandidateFiles(context.Background(), WalkOptions{
		Root:       root,
		StateDir:   filepath.Join(root, ".falkengo"),
		Extensions: []string{".md"},
	})
	if err != nil {
		t.Fatalf("FindCandidateFiles: %v", err)
	}
	got := fileNames(root, files)
	if !reflect.DeepEqual(got, []string{".hidden.md"}) {
		t.Fatalf("files = %v, want explicitly allowed hidden md", got)
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

func writeBytes(t *testing.T, path string, content []byte) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, content, 0o644); err != nil {
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
