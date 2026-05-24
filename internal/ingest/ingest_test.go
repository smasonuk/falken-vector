package ingest

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/vectorstore"
)

func TestRunIndexesAndSkipsUnchangedFiles(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "README.md"), "hello world")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	embedder := &fakeEmbedder{}
	vector := &memoryVectorStore{}
	opts := Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 10,
		Embedder:     embedder,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(10, 0).UTC() },
	}

	first, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("Run first: %v", err)
	}
	if first.NewFiles != 1 || first.ChunksEmbedded != 1 {
		t.Fatalf("first summary = %+v", first)
	}
	second, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("Run second: %v", err)
	}
	if second.UnchangedFiles != 1 || second.ChunksEmbedded != 0 {
		t.Fatalf("second summary = %+v", second)
	}
	if embedder.calls != 1 {
		t.Fatalf("embed calls = %d, want 1", embedder.calls)
	}
}

func TestRunPrintsProgressDuringIngest(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "note.txt"), "abcdefghij")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	vector := &memoryVectorStore{}
	var out bytes.Buffer

	_, err = Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    5,
		ChunkOverlap: 0,
		ChunkerMode:  ChunkerModeFixed,
		Embedder:     &fakeEmbedder{},
		Out:          &out,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(15, 0).UTC() },
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	got := out.String()
	for _, want := range []string{
		"scanning " + root,
		"found 1 candidate files",
		"indexing file 1/1: note.txt",
		"embedding 2 chunks from note.txt with fixed chunker",
		"embedded chunks for note.txt: 1/2",
		"embedded chunks for note.txt: 2/2",
		"committing vector database",
		"updating manifest",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("progress output = %q, want to contain %q", got, want)
		}
	}
}

func TestRunReindexesChangedFileAndSkipsStateDir(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	target := filepath.Join(root, "README.md")
	writeFile(t, target, "hello world")
	writeFile(t, filepath.Join(root, ".falkengo", "ignored.md"), "ignore me")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	embedder := &fakeEmbedder{}
	vector := &memoryVectorStore{}
	opts := Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 10,
		Embedder:     embedder,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(20, 0).UTC() },
	}
	if _, err := Run(ctx, store, opts); err != nil {
		t.Fatalf("Run first: %v", err)
	}
	writeFile(t, target, "changed content")
	second, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("Run changed: %v", err)
	}
	if second.ChangedFiles != 1 || second.Scanned != 1 {
		t.Fatalf("summary = %+v, want one changed scanned file", second)
	}
}

func TestRunSyncSourceMarksMissingFilesDeleted(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	keepPath := filepath.Join(root, "keep.md")
	deletePath := filepath.Join(root, "old.md")
	writeFile(t, keepPath, "keep")
	writeFile(t, deletePath, "old")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	vector := &memoryVectorStore{}
	opts := Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		SyncSource:   true,
		Embedder:     &fakeEmbedder{},
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(50, 0).UTC() },
	}
	if _, err := Run(ctx, store, opts); err != nil {
		t.Fatalf("initial Run: %v", err)
	}
	if err := os.Remove(deletePath); err != nil {
		t.Fatal(err)
	}
	summary, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("sync Run: %v", err)
	}
	if summary.DeletedFiles != 1 || summary.UnchangedFiles != 1 {
		t.Fatalf("summary = %+v, want one deletion and one unchanged", summary)
	}
	deletedDoc, err := store.GetDocumentByPath(ctx, deletePath)
	if err != nil {
		t.Fatalf("GetDocumentByPath deleted: %v", err)
	}
	if deletedDoc.Status != manifest.DocumentStatusDeleted || deletedDoc.DeletedAt == nil {
		t.Fatalf("deleted doc = %+v, want deleted status", deletedDoc)
	}
	oldChunkID := manifest.ChunkID(deletedDoc.ID, 0, HashText("old"))
	active, err := store.GetActiveChunksByIDs(ctx, []string{oldChunkID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs: %v", err)
	}
	if len(active) != 0 {
		t.Fatalf("active old chunks = %+v, want none", active)
	}

	second, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("second sync Run: %v", err)
	}
	if second.DeletedFiles != 0 {
		t.Fatalf("second summary = %+v, want idempotent zero deletions", second)
	}
}

func TestRunSyncSourceDryRunReportsDeletionWithoutWritesOrEmbeddings(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	deletePath := filepath.Join(root, "old.md")
	writeFile(t, deletePath, "old")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	embedder := &fakeEmbedder{}
	vector := &memoryVectorStore{}
	opts := Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		SyncSource:   true,
		Embedder:     embedder,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(60, 0).UTC() },
	}
	if _, err := Run(ctx, store, opts); err != nil {
		t.Fatalf("initial Run: %v", err)
	}
	if err := os.Remove(deletePath); err != nil {
		t.Fatal(err)
	}
	beforeCalls := embedder.calls
	summary, err := Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		SyncSource:   true,
		DryRun:       true,
		Embedder:     embedder,
	})
	if err != nil {
		t.Fatalf("dry-run sync Run: %v", err)
	}
	if summary.DeletedFiles != 1 {
		t.Fatalf("summary = %+v, want one would-delete", summary)
	}
	if embedder.calls != beforeCalls {
		t.Fatalf("embed calls changed from %d to %d during dry-run", beforeCalls, embedder.calls)
	}
	doc, err := store.GetDocumentByPath(ctx, deletePath)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if doc.Status != manifest.DocumentStatusIndexed {
		t.Fatalf("doc status = %q, want still indexed after dry-run", doc.Status)
	}
}

func TestRunSyncSourceOnlyDeletesCurrentSourceRoot(t *testing.T) {
	ctx := context.Background()
	parent := t.TempDir()
	docsRoot := filepath.Join(parent, "docs")
	srcRoot := filepath.Join(parent, "src")
	docsPath := filepath.Join(docsRoot, "doc.md")
	srcPath := filepath.Join(srcRoot, "main.go")
	writeFile(t, docsPath, "docs")
	writeFile(t, srcPath, "package main")
	paths, err := config.ResolvePaths(filepath.Join(parent, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	vector := &memoryVectorStore{}
	base := Options{
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		SyncSource:   true,
		Embedder:     &fakeEmbedder{},
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(70, 0).UTC() },
	}
	docsOpts := base
	docsOpts.Root = docsRoot
	if _, err := Run(ctx, store, docsOpts); err != nil {
		t.Fatalf("docs Run: %v", err)
	}
	srcOpts := base
	srcOpts.Root = srcRoot
	if _, err := Run(ctx, store, srcOpts); err != nil {
		t.Fatalf("src Run: %v", err)
	}
	if err := os.Remove(docsPath); err != nil {
		t.Fatal(err)
	}
	summary, err := Run(ctx, store, srcOpts)
	if err != nil {
		t.Fatalf("src sync Run: %v", err)
	}
	if summary.DeletedFiles != 0 {
		t.Fatalf("src summary = %+v, should not delete docs source", summary)
	}
	doc, err := store.GetDocumentByPath(ctx, docsPath)
	if err != nil {
		t.Fatalf("GetDocumentByPath docs: %v", err)
	}
	if doc.Status != manifest.DocumentStatusIndexed {
		t.Fatalf("docs status = %q, want indexed", doc.Status)
	}
}

func TestRunSyncSourceRecreateAndRename(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	oldPath := filepath.Join(root, "old.md")
	writeFile(t, oldPath, "old")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	vector := &memoryVectorStore{}
	opts := Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		SyncSource:   true,
		Embedder:     &fakeEmbedder{},
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(80, 0).UTC() },
	}
	if _, err := Run(ctx, store, opts); err != nil {
		t.Fatalf("initial Run: %v", err)
	}
	if err := os.Remove(oldPath); err != nil {
		t.Fatal(err)
	}
	if _, err := Run(ctx, store, opts); err != nil {
		t.Fatalf("delete Run: %v", err)
	}
	writeFile(t, oldPath, "recreated")
	recreated, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("recreate Run: %v", err)
	}
	if recreated.ChangedFiles != 1 {
		t.Fatalf("recreate summary = %+v, want changed/reindexed deleted path", recreated)
	}
	doc, err := store.GetDocumentByPath(ctx, oldPath)
	if err != nil {
		t.Fatalf("GetDocumentByPath recreated: %v", err)
	}
	if doc.Status != manifest.DocumentStatusIndexed || doc.DeletedAt != nil {
		t.Fatalf("recreated doc = %+v, want indexed without deleted_at", doc)
	}

	newPath := filepath.Join(root, "new.md")
	if err := os.Rename(oldPath, newPath); err != nil {
		t.Fatal(err)
	}
	renamed, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("rename Run: %v", err)
	}
	if renamed.NewFiles != 1 || renamed.DeletedFiles != 1 {
		t.Fatalf("rename summary = %+v, want old deleted + new indexed", renamed)
	}
	oldDoc, err := store.GetDocumentByPath(ctx, oldPath)
	if err != nil {
		t.Fatalf("GetDocumentByPath old: %v", err)
	}
	if oldDoc.Status != manifest.DocumentStatusDeleted {
		t.Fatalf("old doc status = %q, want deleted", oldDoc.Status)
	}
	newDoc, err := store.GetDocumentByPath(ctx, newPath)
	if err != nil {
		t.Fatalf("GetDocumentByPath new: %v", err)
	}
	if newDoc.Status != manifest.DocumentStatusIndexed {
		t.Fatalf("new doc status = %q, want indexed", newDoc.Status)
	}
}

func TestRunDryRunDoesNotWriteInspectionErrors(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "README.md"), "hello")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := &failingDecisionStore{err: errors.New("manifest unavailable")}
	summary, err := Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		DryRun:       true,
	})
	if err == nil || !strings.Contains(err.Error(), "all files failed") {
		t.Fatalf("Run error = %v, want all files failed", err)
	}
	if summary.FailedFiles != 1 {
		t.Fatalf("summary = %+v, want one failed file", summary)
	}
	if store.markErrorCalled {
		t.Fatal("dry-run called MarkDocumentError")
	}
}

func TestRunAllowsZeroChunkOverlap(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "README.md"), "abcdefghij")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	vector := &memoryVectorStore{}
	summary, err := Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    5,
		ChunkOverlap: 0,
		Embedder:     &fakeEmbedder{},
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(30, 0).UTC() },
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.ChunksEmbedded != 2 {
		t.Fatalf("chunks embedded = %d, want 2", summary.ChunksEmbedded)
	}
}

func TestRunUsesAutoChunkerForMarkdown(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	path := filepath.Join(root, "README.md")
	writeFile(t, path, "# Title\n\nBody\n")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	vector := &memoryVectorStore{}
	_, err = Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		ChunkerMode:  ChunkerModeAuto,
		Embedder:     &fakeEmbedder{},
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(90, 0).UTC() },
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	doc, err := store.GetDocumentByPath(ctx, path)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if doc.Chunker != string(ChunkerModeMarkdown) {
		t.Fatalf("doc chunker = %q, want markdown", doc.Chunker)
	}
	chunkID := manifest.ChunkID(doc.ID, 0, HashText("# Title\n\nBody"))
	chunks, err := store.GetActiveChunksByIDs(ctx, []string{chunkID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs: %v", err)
	}
	if len(chunks) != 1 || chunks[0].Chunker != "markdown" || !sameStrings(chunks[0].HeadingPath, []string{"Title"}) {
		t.Fatalf("chunks = %+v, want markdown heading metadata", chunks)
	}
}

func TestRunEmbedsIndexedTextAndStoresRawTextSeparately(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	path := filepath.Join(root, "README.md")
	raw := "# Title\n\nBody\n"
	writeFile(t, path, raw)
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	embedder := &fakeEmbedder{}
	vector := &memoryVectorStore{}
	_, err = Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		ChunkerMode:  ChunkerModeAuto,
		Embedder:     embedder,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(95, 0).UTC() },
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(embedder.inputs) != 1 {
		t.Fatalf("embed inputs = %+v, want one input", embedder.inputs)
	}
	if embedder.inputs[0] == strings.TrimSpace(raw) || !strings.Contains(embedder.inputs[0], "Document:") || !strings.Contains(embedder.inputs[0], "Section: Title") || !strings.Contains(embedder.inputs[0], "Chunk:\n# Title\n\nBody") {
		t.Fatalf("embed input = %q, want contextual indexed text", embedder.inputs[0])
	}
	doc, err := store.GetDocumentByPath(ctx, path)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if doc.IndexTextVersion != CurrentIndexTextVersion {
		t.Fatalf("doc index text version = %d, want %d", doc.IndexTextVersion, CurrentIndexTextVersion)
	}
	chunkID := manifest.ChunkID(doc.ID, 0, HashText("# Title\n\nBody"))
	chunks, err := store.GetActiveChunksByIDs(ctx, []string{chunkID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if chunks[0].ChunkText != "# Title\n\nBody" {
		t.Fatalf("raw chunk text = %q", chunks[0].ChunkText)
	}
	if chunks[0].IndexedText != embedder.inputs[0] {
		t.Fatalf("indexed text = %q, want embed input %q", chunks[0].IndexedText, embedder.inputs[0])
	}
}

func TestRunReindexesWhenChunkConfigChanges(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "README.md"), "# Title\n\nBody\n")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	embedder := &fakeEmbedder{}
	vector := &memoryVectorStore{}
	opts := Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		ChunkerMode:  ChunkerModeAuto,
		Embedder:     embedder,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(100, 0).UTC() },
	}
	if _, err := Run(ctx, store, opts); err != nil {
		t.Fatalf("Run first: %v", err)
	}
	opts.ChunkerMode = ChunkerModeFixed
	second, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("Run chunker changed: %v", err)
	}
	if second.ChangedFiles != 1 {
		t.Fatalf("second summary = %+v, want changed for chunker change", second)
	}
	opts.ChunkerMode = ChunkerModeFixed
	opts.ChunkSize = 80
	third, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("Run size changed: %v", err)
	}
	if third.ChangedFiles != 1 {
		t.Fatalf("third summary = %+v, want changed for chunk size change", third)
	}
	opts.ChunkOverlap = 10
	fourth, err := Run(ctx, store, opts)
	if err != nil {
		t.Fatalf("Run overlap changed: %v", err)
	}
	if fourth.ChangedFiles != 1 {
		t.Fatalf("fourth summary = %+v, want changed for chunk overlap change", fourth)
	}
}

func TestRunReindexesWhenIndexTextVersionChanges(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	path := filepath.Join(root, "README.md")
	body := "# Title\n\nBody\n"
	writeFile(t, path, body)
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	if err := store.UpsertDocument(ctx, manifest.Document{
		ID:               manifest.DocumentID(path),
		Path:             path,
		ContentHash:      hashString(t, path),
		SizeBytes:        info.Size(),
		ModifiedAt:       info.ModTime().UTC(),
		IndexedAt:        &now,
		Status:           manifest.DocumentStatusIndexed,
		Chunker:          string(ChunkerModeMarkdown),
		ChunkSize:        100,
		ChunkOverlap:     0,
		IndexTextVersion: 0,
	}); err != nil {
		t.Fatalf("UpsertDocument: %v", err)
	}
	embedder := &fakeEmbedder{}
	vector := &memoryVectorStore{}
	summary, err := Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		ChunkerMode:  ChunkerModeAuto,
		Embedder:     embedder,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(110, 0).UTC() },
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if summary.ChangedFiles != 1 || embedder.calls != 1 {
		t.Fatalf("summary = %+v embed calls=%d, want reindex for index text version", summary, embedder.calls)
	}
	doc, err := store.GetDocumentByPath(ctx, path)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if doc.IndexTextVersion != CurrentIndexTextVersion {
		t.Fatalf("index text version = %d, want %d", doc.IndexTextVersion, CurrentIndexTextVersion)
	}
}

func TestChangedFileFailureKeepsPreviousGoodIndexActive(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	target := filepath.Join(root, "README.md")
	writeFile(t, target, "hello world")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	vector := &memoryVectorStore{}
	opts := Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		Embedder:     &fakeEmbedder{},
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(40, 0).UTC() },
	}
	if _, err := Run(ctx, store, opts); err != nil {
		t.Fatalf("initial Run: %v", err)
	}
	doc, err := store.GetDocumentByPath(ctx, target)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	oldChunkID := manifest.ChunkID(doc.ID, 0, HashText("hello world"))

	writeFile(t, target, "changed text")
	opts.Embedder = failingEmbedder{}
	_, err = Run(ctx, store, opts)
	if err == nil || !strings.Contains(err.Error(), "all files failed") {
		t.Fatalf("changed Run error = %v, want all files failed", err)
	}
	doc, err = store.GetDocumentByPath(ctx, target)
	if err != nil {
		t.Fatalf("GetDocumentByPath after failure: %v", err)
	}
	if doc.Status != manifest.DocumentStatusIndexed {
		t.Fatalf("document status = %q, want indexed", doc.Status)
	}
	if doc.Error == nil || !strings.Contains(*doc.Error, "embed failed") {
		t.Fatalf("document error = %v, want embed failure recorded", doc.Error)
	}
	chunks, err := store.GetActiveChunksByIDs(ctx, []string{oldChunkID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs: %v", err)
	}
	if len(chunks) != 1 || chunks[0].ChunkText != "hello world" {
		t.Fatalf("active chunks = %+v, want previous good chunk", chunks)
	}
}

func TestRunRefusesPendingIngestRuns(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "README.md"), "hello")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store := openTestManifest(t, paths.ManifestPath)
	now := time.Now().UTC()
	docID := manifest.DocumentID(filepath.Join(root, "README.md"))
	if err := store.StagePendingDocument(ctx, "run-1", manifest.Document{
		ID:          docID,
		Path:        filepath.Join(root, "README.md"),
		ContentHash: "hash",
		SizeBytes:   5,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      manifest.DocumentStatusIndexed,
	}, nil); err != nil {
		t.Fatalf("StagePendingDocument: %v", err)
	}
	_, err = Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    100,
		ChunkOverlap: 0,
		Embedder:     &fakeEmbedder{},
	})
	if !errors.Is(err, ErrPendingRunDetected) {
		t.Fatalf("Run error = %v, want ErrPendingRunDetected", err)
	}
}

func TestRunStopsOnEmbeddingDeadlineWithoutFailingFileOrSyncing(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	target := filepath.Join(root, "README.md")
	writeFile(t, target, "abcdefghij")
	paths, err := config.ResolvePaths(filepath.Join(root, ".falkengo"))
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	sqliteStore := openTestManifest(t, paths.ManifestPath)
	store := &trackingManifestStore{Store: sqliteStore}
	vector := &memoryVectorStore{}
	embedder := &deadlineAfterFirstEmbedder{}

	summary, err := Run(ctx, store, Options{
		Root:         root,
		Paths:        paths,
		ChunkSize:    5,
		ChunkOverlap: 0,
		ChunkerMode:  ChunkerModeFixed,
		SyncSource:   true,
		Embedder:     embedder,
		OpenVector: func(context.Context, string, int) (vectorstore.Store, error) {
			return vector, nil
		},
		Now: func() time.Time { return time.Unix(120, 0).UTC() },
	})
	if !errors.Is(err, context.DeadlineExceeded) || !strings.Contains(err.Error(), "ingest stopped: context deadline exceeded") {
		t.Fatalf("Run error = %v, want ingest stopped deadline", err)
	}
	if summary.FailedFiles != 0 || summary.ChunksEmbedded != 0 {
		t.Fatalf("summary = %+v, want no failed file or completed chunks", summary)
	}
	if store.listIndexedCalled {
		t.Fatal("ListIndexedDocumentsUnderRoot called after embedding deadline")
	}
	if store.markErrorCalled || store.upsertCalled {
		t.Fatalf("store writes after embedding deadline: markError=%t upsert=%t", store.markErrorCalled, store.upsertCalled)
	}
	if _, err := sqliteStore.GetDocumentByPath(ctx, target); !errors.Is(err, manifest.ErrNotFound) {
		t.Fatalf("GetDocumentByPath error = %v, want no document row after deadline", err)
	}
}

func openTestManifest(t *testing.T, path string) *manifest.SQLiteStore {
	t.Helper()
	if err := config.EnsureStateDirs(config.Paths{StateDir: filepath.Dir(path), LocksPath: filepath.Join(filepath.Dir(path), "locks")}); err != nil {
		t.Fatalf("EnsureStateDirs: %v", err)
	}
	store, err := manifest.Open(path)
	if err != nil {
		t.Fatalf("manifest.Open: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if err := store.Init(context.Background()); err != nil {
		t.Fatalf("Init: %v", err)
	}
	return store
}

func hashString(t *testing.T, path string) string {
	t.Helper()
	hash, err := HashFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return hash
}

type fakeEmbedder struct {
	calls  int
	inputs []string
}

func (e *fakeEmbedder) EmbedText(_ context.Context, input string) (llm.Embedding, error) {
	e.calls++
	e.inputs = append(e.inputs, input)
	return llm.Embedding{Model: "fake", Vector: []float32{float32(len(input)), 1, 0}}, nil
}

type failingEmbedder struct{}

func (failingEmbedder) EmbedText(context.Context, string) (llm.Embedding, error) {
	return llm.Embedding{}, errors.New("embed failed")
}

type deadlineAfterFirstEmbedder struct {
	calls int
}

func (e *deadlineAfterFirstEmbedder) EmbedText(_ context.Context, input string) (llm.Embedding, error) {
	e.calls++
	if e.calls > 1 {
		return llm.Embedding{}, context.DeadlineExceeded
	}
	return llm.Embedding{Model: "fake", Vector: []float32{float32(len(input)), 1, 0}}, nil
}

type failingDecisionStore struct {
	manifest.EmptyStore
	err             error
	markErrorCalled bool
}

func (s *failingDecisionStore) GetDocumentByPath(context.Context, string) (*manifest.Document, error) {
	return nil, s.err
}

func (s *failingDecisionStore) MarkDocumentError(context.Context, string, string) error {
	s.markErrorCalled = true
	return nil
}

type trackingManifestStore struct {
	manifest.Store
	listIndexedCalled bool
	markErrorCalled   bool
	upsertCalled      bool
}

func (s *trackingManifestStore) ListIndexedDocumentsUnderRoot(ctx context.Context, sourceRoot string) ([]manifest.Document, error) {
	s.listIndexedCalled = true
	return s.Store.ListIndexedDocumentsUnderRoot(ctx, sourceRoot)
}

func (s *trackingManifestStore) MarkDocumentError(ctx context.Context, path string, errText string) error {
	s.markErrorCalled = true
	return s.Store.MarkDocumentError(ctx, path, errText)
}

func (s *trackingManifestStore) UpsertDocument(ctx context.Context, doc manifest.Document) error {
	s.upsertCalled = true
	return s.Store.UpsertDocument(ctx, doc)
}

type memoryVectorStore struct {
	ids []string
}

func (s *memoryVectorStore) Close() error { return nil }
func (s *memoryVectorStore) Insert(_ context.Context, _ []float32, chunkID string) (int64, error) {
	s.ids = append(s.ids, chunkID)
	return int64(len(s.ids)), nil
}
func (s *memoryVectorStore) Search(context.Context, []float32, int) ([]vectorstore.Hit, error) {
	return nil, nil
}
func (s *memoryVectorStore) Commit(context.Context) error { return nil }
