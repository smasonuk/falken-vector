package falkenvector

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/pkg/embeddings"
)

func TestErrNoIndexMatchesInternalNoIndex(t *testing.T) {
	engine, err := NewEngine(EngineConfig{StateDir: filepath.Join(t.TempDir(), "state")})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	if _, err := engine.Status(context.Background()); !errors.Is(err, ErrNoIndex) {
		t.Fatalf("Status error = %v, want ErrNoIndex", err)
	}
	if _, err := engine.Query(context.Background(), QueryRequest{Question: "missing"}); !errors.Is(err, ErrNoIndex) {
		t.Fatalf("Query error = %v, want ErrNoIndex", err)
	}
}

func TestPublicIngestTypesConstruct(t *testing.T) {
	request := IngestRequest{
		Root:              ".",
		Extensions:        []string{"go", "md"},
		ExcludeExtensions: []string{"tmp"},
		ExcludeDirs:       []string{"fixtures"},
		ChunkSize:         1200,
		ChunkOverlap:      200,
		ChunkerMode:       ChunkerAuto,
		DryRun:            true,
		SyncSource:        true,
	}
	result := IngestResult{
		Directory:      "/repo",
		Scanned:        1,
		NewFiles:       1,
		ChunksEmbedded: 0,
		Warnings:       []string{"note"},
	}
	if request.ChunkerMode != ChunkerAuto || len(request.ExcludeExtensions) != 1 || len(request.ExcludeDirs) != 1 || result.Directory == "" || len(result.Warnings) != 1 {
		t.Fatalf("request=%+v result=%+v", request, result)
	}
}

func TestIngestRequestDefaultsMatchCLIIngestDefaults(t *testing.T) {
	request := normalizeIngestRequest(IngestRequest{})
	if request.Root != "." {
		t.Fatalf("Root default = %q, want .", request.Root)
	}
	if request.ChunkerMode != ChunkerAuto {
		t.Fatalf("ChunkerMode default = %q, want auto", request.ChunkerMode)
	}
	if request.ChunkSize != 0 {
		t.Fatalf("ChunkSize default = %d, want internal default sentinel", request.ChunkSize)
	}
	if request.ChunkOverlap != 200 {
		t.Fatalf("ChunkOverlap default = %d, want CLI default 200", request.ChunkOverlap)
	}
	if request.Extensions != nil {
		t.Fatalf("Extensions default = %+v, want nil", request.Extensions)
	}
	if request.ExcludeExtensions != nil {
		t.Fatalf("ExcludeExtensions default = %+v, want nil", request.ExcludeExtensions)
	}
	if request.ExcludeDirs != nil {
		t.Fatalf("ExcludeDirs default = %+v, want nil", request.ExcludeDirs)
	}
}

func TestEngineIngestDryRunNoManifestDoesNotWriteState(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("# Title\n\nAlpha token.\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	stateDir := filepath.Join(t.TempDir(), "state")
	engine, err := NewEngine(EngineConfig{StateDir: stateDir})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var result IngestResult
	stdout, stderr := captureProcessOutput(t, func() {
		var err error
		result, err = engine.Ingest(context.Background(), IngestRequest{Root: root, DryRun: true})
		if err != nil {
			t.Fatalf("Ingest: %v", err)
		}
	})
	if stdout != "" || stderr != "" {
		t.Fatalf("stdout=%q stderr=%q, want silent", stdout, stderr)
	}
	if result.Directory != root || result.Scanned != 1 || result.NewFiles != 1 || result.ChunksEmbedded != 0 {
		t.Fatalf("result = %+v, want dry-run scan of one new file", result)
	}
	if _, err := os.Stat(stateDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("state dir stat error = %v, want not exist", err)
	}
}

func TestEngineIngestDryRunRespectsExcludes(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "keep.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatalf("write keep: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "skip.md"), []byte("# Skip\n"), 0o644); err != nil {
		t.Fatalf("write skip: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(root, "fixtures"), 0o755); err != nil {
		t.Fatalf("mkdir fixtures: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "fixtures", "fixture.go"), []byte("package fixture\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	stateDir := filepath.Join(t.TempDir(), "state")
	engine, err := NewEngine(EngineConfig{StateDir: stateDir})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	result, err := engine.Ingest(context.Background(), IngestRequest{
		Root:              root,
		DryRun:            true,
		Extensions:        []string{"go", "md"},
		ExcludeExtensions: []string{"md"},
		ExcludeDirs:       []string{"fixtures"},
	})
	if err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	if result.Scanned != 1 || result.NewFiles != 1 {
		t.Fatalf("result = %+v, want only keep.go", result)
	}
	if _, err := os.Stat(stateDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("state dir stat error = %v, want not exist", err)
	}
}

func TestEngineIngestRejectsInvalidChunker(t *testing.T) {
	var events []Event
	engine, err := NewEngine(EngineConfig{
		StateDir: filepath.Join(t.TempDir(), "state"),
		Events: func(event Event) {
			events = append(events, event)
		},
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	_, err = engine.Ingest(context.Background(), IngestRequest{Root: t.TempDir(), DryRun: true, ChunkerMode: ChunkerMode("bogus")})
	if err == nil || !strings.Contains(err.Error(), "--chunker must be auto, fixed, markdown, text, or code") {
		t.Fatalf("Ingest error = %v, want chunker validation", err)
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, []EventType{EventRunStarted, EventRunFailed}) {
		t.Fatalf("events = %v, want run start/fail", got)
	}
}

func TestEngineIngestRealRequiresEmbeddingConfig(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "note.md"), []byte("Alpha token.\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	engine, err := NewEngine(EngineConfig{StateDir: filepath.Join(t.TempDir(), "state")})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	_, err = engine.Ingest(context.Background(), IngestRequest{Root: root})
	if !errors.Is(err, embeddings.ErrBaseURLRequired) {
		t.Fatalf("Ingest error = %v, want base URL required", err)
	}
	if !strings.Contains(err.Error(), "configure embedder") {
		t.Fatalf("Ingest error = %v, want configure embedder context", err)
	}
}

func TestEngineIngestEmitsRunEvents(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "note.md"), []byte("Alpha token.\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	var events []Event
	engine, err := NewEngine(EngineConfig{
		StateDir: filepath.Join(t.TempDir(), "state"),
		Embedding: ModelConfig{
			BaseURL: "https://embed.test/v1",
			Model:   "embed-model",
		},
		HTTPClient: fakeHTTPClient{body: `{"model":"embed-model","data":[{"embedding":[1,0,0]}],"usage":{}}`},
		Events: func(event Event) {
			events = append(events, event)
		},
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	result, err := engine.Ingest(context.Background(), IngestRequest{Root: root})
	if err != nil {
		t.Fatalf("Ingest: %v", err)
	}
	if result.Scanned != 1 || result.NewFiles != 1 || result.ChunksEmbedded == 0 {
		t.Fatalf("result = %+v, want real ingest summary", result)
	}
	types := eventTypes(events)
	if len(types) == 0 || types[0] != EventRunStarted || types[len(types)-1] != EventRunCompleted {
		t.Fatalf("events = %v, want run started first and completed last", types)
	}
	for _, want := range []EventType{EventIngestStarted, EventEmbeddingRequest, EventEmbeddingResponse, EventIngestCompleted} {
		if !hasEventType(types, want) {
			t.Fatalf("events = %v, want %s", types, want)
		}
	}
}

func hasEventType(types []EventType, want EventType) bool {
	for _, got := range types {
		if got == want {
			return true
		}
	}
	return false
}
