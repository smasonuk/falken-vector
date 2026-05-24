package ingest

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/smasonuk/falken-vector/internal/manifest"
)

func TestHashFileStable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.txt")
	writeFile(t, path, "same")
	first, err := HashFile(path)
	if err != nil {
		t.Fatalf("HashFile: %v", err)
	}
	second, err := HashFile(path)
	if err != nil {
		t.Fatalf("HashFile second: %v", err)
	}
	if first != second {
		t.Fatalf("hashes differ: %q %q", first, second)
	}
}

func TestDecideFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.txt")
	writeFile(t, path, "same")
	hash, err := HashFile(path)
	if err != nil {
		t.Fatal(err)
	}
	candidate := CandidateFile{Path: path, ModifiedAt: time.Now(), SizeBytes: 4}

	store := &decisionStore{docs: map[string]*manifest.Document{}}
	decision, _, err := DecideFile(context.Background(), store, candidate)
	if err != nil {
		t.Fatalf("DecideFile new: %v", err)
	}
	if decision != FileDecisionNew {
		t.Fatalf("decision = %q, want new", decision)
	}

	store.docs[path] = &manifest.Document{Path: path, ContentHash: hash, Status: manifest.DocumentStatusIndexed}
	decision, _, err = DecideFile(context.Background(), store, candidate)
	if err != nil {
		t.Fatalf("DecideFile unchanged: %v", err)
	}
	if decision != FileDecisionUnchanged {
		t.Fatalf("decision = %q, want unchanged", decision)
	}

	writeFile(t, path, "changed")
	decision, _, err = DecideFile(context.Background(), store, candidate)
	if err != nil {
		t.Fatalf("DecideFile changed: %v", err)
	}
	if decision != FileDecisionChanged {
		t.Fatalf("decision = %q, want changed", decision)
	}

	newHash, _ := HashFile(path)
	store.docs[path] = &manifest.Document{Path: path, ContentHash: newHash, Status: manifest.DocumentStatusError}
	decision, _, err = DecideFile(context.Background(), store, candidate)
	if err != nil {
		t.Fatalf("DecideFile previous error: %v", err)
	}
	if decision != FileDecisionChanged {
		t.Fatalf("decision = %q, want retry changed", decision)
	}
}

func TestDecideFileReindexesWhenChunkConfigChanges(t *testing.T) {
	path := filepath.Join(t.TempDir(), "a.md")
	writeFile(t, path, "same")
	hash, err := HashFile(path)
	if err != nil {
		t.Fatal(err)
	}
	candidate := CandidateFile{Path: path, ModifiedAt: time.Now(), SizeBytes: 4}
	store := &decisionStore{docs: map[string]*manifest.Document{
		path: {
			Path:             path,
			ContentHash:      hash,
			Status:           manifest.DocumentStatusIndexed,
			Chunker:          "markdown",
			ChunkSize:        1200,
			ChunkOverlap:     200,
			IndexTextVersion: CurrentIndexTextVersion,
		},
	}}
	decision, _, err := DecideFileWithConfig(context.Background(), store, candidate, ChunkConfig{Chunker: "markdown", ChunkSize: 1200, ChunkOverlap: 200, IndexTextVersion: CurrentIndexTextVersion})
	if err != nil {
		t.Fatalf("DecideFileWithConfig unchanged: %v", err)
	}
	if decision != FileDecisionUnchanged {
		t.Fatalf("decision = %q, want unchanged", decision)
	}
	decision, _, err = DecideFileWithConfig(context.Background(), store, candidate, ChunkConfig{Chunker: "fixed", ChunkSize: 1200, ChunkOverlap: 200, IndexTextVersion: CurrentIndexTextVersion})
	if err != nil {
		t.Fatalf("DecideFileWithConfig chunker changed: %v", err)
	}
	if decision != FileDecisionChanged {
		t.Fatalf("decision = %q, want changed for chunker change", decision)
	}
	decision, _, err = DecideFileWithConfig(context.Background(), store, candidate, ChunkConfig{Chunker: "markdown", ChunkSize: 800, ChunkOverlap: 200, IndexTextVersion: CurrentIndexTextVersion})
	if err != nil {
		t.Fatalf("DecideFileWithConfig size changed: %v", err)
	}
	if decision != FileDecisionChanged {
		t.Fatalf("decision = %q, want changed for size change", decision)
	}
	decision, _, err = DecideFileWithConfig(context.Background(), store, candidate, ChunkConfig{Chunker: "markdown", ChunkSize: 1200, ChunkOverlap: 200, IndexTextVersion: CurrentIndexTextVersion + 1})
	if err != nil {
		t.Fatalf("DecideFileWithConfig index text version changed: %v", err)
	}
	if decision != FileDecisionChanged {
		t.Fatalf("decision = %q, want changed for index text version change", decision)
	}
}

type decisionStore struct {
	manifest.Store
	docs map[string]*manifest.Document
}

func (s *decisionStore) GetDocumentByPath(_ context.Context, path string) (*manifest.Document, error) {
	doc, ok := s.docs[path]
	if !ok {
		return nil, manifest.ErrNotFound
	}
	return doc, nil
}

func (s *decisionStore) Close() error               { return nil }
func (s *decisionStore) Init(context.Context) error { return nil }
func (s *decisionStore) GetDocumentByID(context.Context, string) (*manifest.Document, error) {
	return nil, errors.New("unused")
}
func (s *decisionStore) UpsertDocument(context.Context, manifest.Document) error { return nil }
func (s *decisionStore) MarkDocumentError(context.Context, string, string) error { return nil }
func (s *decisionStore) MarkChunksInactive(context.Context, string) error        { return nil }
func (s *decisionStore) InsertChunk(context.Context, manifest.Chunk) error       { return nil }
func (s *decisionStore) GetActiveChunksByIDs(context.Context, []string) ([]manifest.Chunk, error) {
	return nil, nil
}
func (s *decisionStore) Stats(context.Context) (manifest.Stats, error) { return manifest.Stats{}, nil }
