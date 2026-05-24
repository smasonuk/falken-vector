package vectorstore

import (
	"context"
	"testing"
)

func TestVecgoInsertCommitSearch(t *testing.T) {
	ctx := context.Background()
	store, err := Open(ctx, t.TempDir(), 3)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer store.Close()

	if _, err := store.Insert(ctx, []float32{1, 0, 0}, "chunk-1"); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if err := store.Commit(ctx); err != nil {
		t.Fatalf("Commit: %v", err)
	}
	hits, err := store.Search(ctx, []float32{1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(hits) != 1 || hits[0].ChunkID != "chunk-1" {
		t.Fatalf("hits = %+v, want chunk-1", hits)
	}
}
