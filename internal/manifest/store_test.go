package manifest

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestInitCreatesSchemaAndPreservesData(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "manifest.sqlite")
	store, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := store.Init(ctx); err != nil {
		t.Fatalf("Init: %v", err)
	}
	now := time.Now().UTC()
	doc := Document{
		ID:          DocumentID("/tmp/readme.md"),
		Path:        "/tmp/readme.md",
		ContentHash: "abc",
		SizeBytes:   12,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      DocumentStatusIndexed,
	}
	if err := store.UpsertDocument(ctx, doc); err != nil {
		t.Fatalf("UpsertDocument: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	store, err = Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer store.Close()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("second Init: %v", err)
	}
	got, err := store.GetDocumentByPath(ctx, doc.Path)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if got.ContentHash != doc.ContentHash || got.Status != DocumentStatusIndexed {
		t.Fatalf("document = %+v, want preserved data", got)
	}
}

func TestDocumentCanBeInsertedAndFetchedByPath(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	doc := Document{
		ID:          DocumentID("/tmp/a.go"),
		Path:        "/tmp/a.go",
		ContentHash: "hash",
		SizeBytes:   99,
		ModifiedAt:  now,
		Status:      DocumentStatusIndexed,
	}
	if err := store.UpsertDocument(ctx, doc); err != nil {
		t.Fatalf("UpsertDocument: %v", err)
	}
	got, err := store.GetDocumentByPath(ctx, doc.Path)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if got.ID != doc.ID || got.Path != doc.Path || got.ContentHash != doc.ContentHash {
		t.Fatalf("document = %+v, want %+v", got, doc)
	}
}

func TestPreMigrationDocumentReadsWithChunkConfigDefaults(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "manifest.sqlite")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open raw sqlite: %v", err)
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	if _, err := db.ExecContext(ctx, `
create table documents (
  id text primary key,
  path text not null unique,
  content_hash text not null,
  size_bytes integer not null,
  modified_at text not null,
  indexed_at text,
  status text not null,
  error text
);
create table chunks (
  id text primary key,
  document_id text not null,
  chunk_index integer not null,
  content_hash text not null,
  chunk_text text not null,
  start_line integer,
  end_line integer,
  vector_id integer,
  embedding_model text not null,
  active integer not null default 1,
  created_at text not null
);
`); err != nil {
		db.Close()
		t.Fatalf("create pre-migration manifest: %v", err)
	}
	docID := DocumentID("/tmp/old.md")
	chunkID := ChunkID(docID, 0, "chunk")
	if _, err := db.ExecContext(ctx, `
insert into documents(id, path, content_hash, size_bytes, modified_at, indexed_at, status, error)
values (?, ?, ?, ?, ?, ?, ?, null);
`, docID, "/tmp/old.md", "hash", 12, now, now, DocumentStatusIndexed); err != nil {
		db.Close()
		t.Fatalf("insert pre-migration document: %v", err)
	}
	if _, err := db.ExecContext(ctx, `
insert into chunks(id, document_id, chunk_index, content_hash, chunk_text, start_line, end_line, embedding_model, active, created_at)
values (?, ?, ?, ?, ?, 1, 2, ?, 1, ?);
`, chunkID, docID, 0, "chunk", "legacy chunk text", "model", now); err != nil {
		db.Close()
		t.Fatalf("insert pre-migration chunk: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("close raw sqlite: %v", err)
	}

	store, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer store.Close()

	doc, err := store.GetDocumentByPath(ctx, "/tmp/old.md")
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if doc.Chunker != "" || doc.ChunkSize != 0 || doc.ChunkOverlap != 0 || doc.IndexTextVersion != 0 || doc.SourceRoot != "" || doc.DeletedAt != nil {
		t.Fatalf("pre-migration defaults = %+v", doc)
	}
	chunks, err := store.GetActiveChunksByIDs(ctx, []string{chunkID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs: %v", err)
	}
	if len(chunks) != 1 || chunks[0].ChunkText != "legacy chunk text" || chunks[0].IndexedText != "" {
		t.Fatalf("pre-migration chunks = %+v", chunks)
	}
}

func TestChunkMetadataRoundTrips(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/tmp/readme.md")
	doc := Document{
		ID:               docID,
		Path:             "/tmp/readme.md",
		ContentHash:      "hash",
		SizeBytes:        99,
		ModifiedAt:       now,
		IndexedAt:        &now,
		Status:           DocumentStatusIndexed,
		Chunker:          "markdown",
		ChunkSize:        1200,
		ChunkOverlap:     200,
		IndexTextVersion: 1,
	}
	chunk := Chunk{
		ID:             ChunkID(docID, 0, "chunkhash"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunkhash",
		ChunkText:      "# Title\nbody",
		IndexedText:    "Document: /tmp/readme.md\nSection: Title > Install\nChunk:\n# Title\nbody",
		StartLine:      1,
		EndLine:        2,
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
		Chunker:        "markdown",
		Language:       "",
		HeadingPath:    []string{"Title", "Install"},
		SymbolName:     "",
		SymbolKind:     "",
	}
	if err := store.ReplaceDocumentChunks(ctx, doc, []Chunk{chunk}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
	gotDoc, err := store.GetDocumentByPath(ctx, doc.Path)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if gotDoc.Chunker != "markdown" || gotDoc.ChunkSize != 1200 || gotDoc.ChunkOverlap != 200 || gotDoc.IndexTextVersion != 1 {
		t.Fatalf("document chunk config = %+v", gotDoc)
	}
	chunks, err := store.GetActiveChunksByIDs(ctx, []string{chunk.ID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	got := chunks[0]
	if got.Chunker != chunk.Chunker || got.IndexedText != chunk.IndexedText || !sameStringSlice(got.HeadingPath, chunk.HeadingPath) {
		t.Fatalf("chunk metadata = %+v, want %+v", got, chunk)
	}
}

func TestChunksCanBeMarkedInactive(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/tmp/a.go")
	if err := store.UpsertDocument(ctx, Document{
		ID:          docID,
		Path:        "/tmp/a.go",
		ContentHash: "hash",
		SizeBytes:   99,
		ModifiedAt:  now,
		Status:      DocumentStatusIndexed,
	}); err != nil {
		t.Fatalf("UpsertDocument: %v", err)
	}
	vectorID := int64(7)
	chunk := Chunk{
		ID:             ChunkID(docID, 0, "chunkhash"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunkhash",
		ChunkText:      "hello",
		StartLine:      1,
		EndLine:        1,
		VectorID:       &vectorID,
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}
	if err := store.InsertChunk(ctx, chunk); err != nil {
		t.Fatalf("InsertChunk: %v", err)
	}
	if err := store.MarkChunksInactive(ctx, docID); err != nil {
		t.Fatalf("MarkChunksInactive: %v", err)
	}
	chunks, err := store.GetActiveChunksByIDs(ctx, []string{chunk.ID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs: %v", err)
	}
	if len(chunks) != 0 {
		t.Fatalf("active chunks = %d, want 0", len(chunks))
	}
	stats, err := store.Stats(ctx)
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if stats.InactiveChunks != 1 {
		t.Fatalf("InactiveChunks = %d, want 1", stats.InactiveChunks)
	}
}

func TestPendingRunStagesBeforeActivation(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/tmp/a.go")
	oldVectorID := int64(1)
	oldChunk := Chunk{
		ID:             ChunkID(docID, 0, "old"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "old",
		ChunkText:      "old text",
		StartLine:      1,
		EndLine:        1,
		VectorID:       &oldVectorID,
		EmbeddingModel: "old-model",
		Active:         true,
		CreatedAt:      now,
	}
	indexedAt := now
	doc := Document{
		ID:          docID,
		Path:        "/tmp/a.go",
		ContentHash: "old-doc",
		SizeBytes:   8,
		ModifiedAt:  now,
		IndexedAt:   &indexedAt,
		Status:      DocumentStatusIndexed,
	}
	if err := store.ReplaceDocumentChunks(ctx, doc, []Chunk{oldChunk}); err != nil {
		t.Fatalf("ReplaceDocumentChunks old: %v", err)
	}

	newVectorID := int64(2)
	newDoc := doc
	newDoc.ContentHash = "new-doc"
	newDoc.Chunker = "code"
	newDoc.ChunkSize = 900
	newDoc.ChunkOverlap = 100
	newDoc.IndexTextVersion = 1
	newChunk := oldChunk
	newChunk.ID = ChunkID(docID, 0, "new")
	newChunk.ContentHash = "new"
	newChunk.ChunkText = "new text"
	newChunk.IndexedText = "File: /tmp/a.go\nLanguage: go\nSymbol: function Run\nChunk:\nnew text"
	newChunk.VectorID = &newVectorID
	newChunk.Chunker = "code"
	newChunk.Language = "go"
	newChunk.SymbolName = "Run"
	newChunk.SymbolKind = "function"
	if err := store.StagePendingDocument(ctx, "run-1", newDoc, []Chunk{newChunk}); err != nil {
		t.Fatalf("StagePendingDocument: %v", err)
	}
	runs, err := store.ListPendingRuns(ctx)
	if err != nil {
		t.Fatalf("ListPendingRuns: %v", err)
	}
	if len(runs) != 1 || runs[0].ID != "run-1" || runs[0].DocumentCount != 1 || runs[0].ChunkCount != 1 {
		t.Fatalf("pending runs = %+v, want run-1 with counts", runs)
	}

	active, err := store.GetActiveChunksByIDs(ctx, []string{oldChunk.ID, newChunk.ID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs before activate: %v", err)
	}
	if len(active) != 1 || active[0].ID != oldChunk.ID {
		t.Fatalf("active before activate = %+v, want old chunk only", active)
	}

	if err := store.ActivatePendingRun(ctx, "run-1"); err != nil {
		t.Fatalf("ActivatePendingRun: %v", err)
	}
	active, err = store.GetActiveChunksByIDs(ctx, []string{oldChunk.ID, newChunk.ID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs after activate: %v", err)
	}
	if len(active) != 1 || active[0].ID != newChunk.ID || active[0].ChunkText != "new text" || active[0].IndexedText != newChunk.IndexedText || active[0].Language != "go" || active[0].SymbolName != "Run" {
		t.Fatalf("active after activate = %+v, want new chunk only", active)
	}
	gotDoc, err := store.GetDocumentByPath(ctx, newDoc.Path)
	if err != nil {
		t.Fatalf("GetDocumentByPath after activate: %v", err)
	}
	if gotDoc.Chunker != "code" || gotDoc.ChunkSize != 900 || gotDoc.ChunkOverlap != 100 || gotDoc.IndexTextVersion != 1 {
		t.Fatalf("document chunk config after activate = %+v", gotDoc)
	}
	runs, err = store.ListPendingRuns(ctx)
	if err != nil {
		t.Fatalf("ListPendingRuns after activate: %v", err)
	}
	if len(runs) != 0 {
		t.Fatalf("pending runs after activate = %+v, want none", runs)
	}
}

func TestClearPendingRunRemovesStagedRows(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/tmp/a.go")
	vectorID := int64(2)
	doc := Document{
		ID:          docID,
		Path:        "/tmp/a.go",
		ContentHash: "doc",
		SizeBytes:   8,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      DocumentStatusIndexed,
	}
	chunk := Chunk{
		ID:             ChunkID(docID, 0, "chunk"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk",
		ChunkText:      "text",
		VectorID:       &vectorID,
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}
	if err := store.StagePendingDocument(ctx, "run-1", doc, []Chunk{chunk}); err != nil {
		t.Fatalf("StagePendingDocument: %v", err)
	}
	if err := store.ClearPendingRun(ctx, "run-1"); err != nil {
		t.Fatalf("ClearPendingRun: %v", err)
	}
	runs, err := store.ListPendingRuns(ctx)
	if err != nil {
		t.Fatalf("ListPendingRuns: %v", err)
	}
	if len(runs) != 0 {
		t.Fatalf("pending runs = %+v, want none", runs)
	}
}

func TestMarkDocumentErrorPreservesIndexedDocument(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	indexedAt := now
	doc := Document{
		ID:          DocumentID("/tmp/a.go"),
		Path:        "/tmp/a.go",
		ContentHash: "good",
		SizeBytes:   10,
		ModifiedAt:  now,
		IndexedAt:   &indexedAt,
		Status:      DocumentStatusIndexed,
	}
	if err := store.UpsertDocument(ctx, doc); err != nil {
		t.Fatalf("UpsertDocument: %v", err)
	}
	if err := store.MarkDocumentError(ctx, doc.Path, "latest failure"); err != nil {
		t.Fatalf("MarkDocumentError: %v", err)
	}
	got, err := store.GetDocumentByPath(ctx, doc.Path)
	if err != nil {
		t.Fatalf("GetDocumentByPath: %v", err)
	}
	if got.Status != DocumentStatusIndexed || got.ContentHash != "good" {
		t.Fatalf("document = %+v, want indexed good document preserved", got)
	}
	if got.Error == nil || *got.Error != "latest failure" {
		t.Fatalf("document error = %v, want recorded latest failure", got.Error)
	}
}

func TestMarkDocumentDeletedMarksChunksInactive(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	indexedAt := now
	docID := DocumentID("/tmp/a.go")
	vectorID := int64(9)
	doc := Document{
		ID:          docID,
		Path:        "/tmp/a.go",
		ContentHash: "hash",
		SizeBytes:   10,
		ModifiedAt:  now,
		IndexedAt:   &indexedAt,
		SourceRoot:  "/tmp",
		Status:      DocumentStatusIndexed,
	}
	chunk := Chunk{
		ID:             ChunkID(docID, 0, "chunk"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk",
		ChunkText:      "text",
		VectorID:       &vectorID,
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}
	if err := store.ReplaceDocumentChunks(ctx, doc, []Chunk{chunk}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
	deletedAt := now.Add(time.Minute)
	if err := store.MarkDocumentDeleted(ctx, docID, deletedAt); err != nil {
		t.Fatalf("MarkDocumentDeleted: %v", err)
	}
	got, err := store.GetDocumentByID(ctx, docID)
	if err != nil {
		t.Fatalf("GetDocumentByID: %v", err)
	}
	if got.Status != DocumentStatusDeleted || got.IndexedAt != nil || got.DeletedAt == nil {
		t.Fatalf("document = %+v, want deleted with deleted_at", got)
	}
	active, err := store.GetActiveChunksByIDs(ctx, []string{chunk.ID})
	if err != nil {
		t.Fatalf("GetActiveChunksByIDs: %v", err)
	}
	if len(active) != 0 {
		t.Fatalf("active chunks = %+v, want none", active)
	}
	stats, err := store.Stats(ctx)
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if stats.DeletedDocuments != 1 || stats.ActiveChunks != 0 || stats.InactiveChunks != 1 {
		t.Fatalf("stats = %+v, want deleted/inactive counts", stats)
	}
}

func TestListIndexedDocumentsUnderRootScopesByRoot(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docs := []Document{
		{ID: DocumentID("/docs/a.md"), Path: "/docs/a.md", ContentHash: "a", SizeBytes: 1, ModifiedAt: now, SourceRoot: "/docs", Status: DocumentStatusIndexed},
		{ID: DocumentID("/src/b.go"), Path: "/src/b.go", ContentHash: "b", SizeBytes: 1, ModifiedAt: now, SourceRoot: "/src", Status: DocumentStatusIndexed},
		{ID: DocumentID("/docs/deleted.md"), Path: "/docs/deleted.md", ContentHash: "c", SizeBytes: 1, ModifiedAt: now, SourceRoot: "/docs", Status: DocumentStatusDeleted},
	}
	for _, doc := range docs {
		if err := store.UpsertDocument(ctx, doc); err != nil {
			t.Fatalf("UpsertDocument: %v", err)
		}
	}
	got, err := store.ListIndexedDocumentsUnderRoot(ctx, "/docs")
	if err != nil {
		t.Fatalf("ListIndexedDocumentsUnderRoot: %v", err)
	}
	if len(got) != 1 || got[0].Path != "/docs/a.md" {
		t.Fatalf("docs = %+v, want only indexed /docs document", got)
	}
}

func TestListActiveIndexedChunksFiltersAndOrders(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	indexedAt := now
	docs := []Document{
		{ID: DocumentID("/tmp/b.md"), Path: "/tmp/b.md", ContentHash: "b", SizeBytes: 1, ModifiedAt: now, IndexedAt: &indexedAt, Status: DocumentStatusIndexed},
		{ID: DocumentID("/tmp/a.md"), Path: "/tmp/a.md", ContentHash: "a", SizeBytes: 1, ModifiedAt: now, IndexedAt: &indexedAt, Status: DocumentStatusIndexed},
		{ID: DocumentID("/tmp/error.md"), Path: "/tmp/error.md", ContentHash: "e", SizeBytes: 1, ModifiedAt: now, Status: DocumentStatusError},
		{ID: DocumentID("/tmp/deleted.md"), Path: "/tmp/deleted.md", ContentHash: "d", SizeBytes: 1, ModifiedAt: now, Status: DocumentStatusDeleted},
	}
	for _, doc := range docs {
		if err := store.UpsertDocument(ctx, doc); err != nil {
			t.Fatalf("UpsertDocument %s: %v", doc.Path, err)
		}
	}
	vectorID := int64(1)
	chunks := []Chunk{
		{ID: ChunkID(docs[0].ID, 0, "b0"), DocumentID: docs[0].ID, ChunkIndex: 0, ContentHash: "b0", ChunkText: "b", VectorID: &vectorID, EmbeddingModel: "old", Active: true, CreatedAt: now},
		{ID: ChunkID(docs[1].ID, 0, "a0"), DocumentID: docs[1].ID, ChunkIndex: 0, ContentHash: "a0", ChunkText: "a", VectorID: &vectorID, EmbeddingModel: "old", Active: true, CreatedAt: now},
		{ID: ChunkID(docs[1].ID, 1, "a1"), DocumentID: docs[1].ID, ChunkIndex: 1, ContentHash: "a1", ChunkText: "inactive", VectorID: &vectorID, EmbeddingModel: "old", Active: false, CreatedAt: now},
		{ID: ChunkID(docs[2].ID, 0, "e0"), DocumentID: docs[2].ID, ChunkIndex: 0, ContentHash: "e0", ChunkText: "error", VectorID: &vectorID, EmbeddingModel: "old", Active: true, CreatedAt: now},
		{ID: ChunkID(docs[3].ID, 0, "d0"), DocumentID: docs[3].ID, ChunkIndex: 0, ContentHash: "d0", ChunkText: "deleted", VectorID: &vectorID, EmbeddingModel: "old", Active: true, CreatedAt: now},
	}
	for _, chunk := range chunks {
		if err := store.InsertChunk(ctx, chunk); err != nil {
			t.Fatalf("InsertChunk %s: %v", chunk.ID, err)
		}
	}

	got, err := store.ListActiveIndexedChunks(ctx, 10, 0)
	if err != nil {
		t.Fatalf("ListActiveIndexedChunks: %v", err)
	}
	if len(got) != 2 || got[0].ID != chunks[1].ID || got[1].ID != chunks[0].ID {
		t.Fatalf("chunks = %+v, want active indexed chunks ordered by document path", got)
	}
	page, err := store.ListActiveIndexedChunks(ctx, 1, 1)
	if err != nil {
		t.Fatalf("ListActiveIndexedChunks page: %v", err)
	}
	if len(page) != 1 || page[0].ID != chunks[0].ID {
		t.Fatalf("page = %+v, want second ordered chunk", page)
	}
}

func TestUpdateChunkVectorRefsRequiresActiveChunk(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/tmp/a.md")
	if err := store.UpsertDocument(ctx, Document{
		ID:          docID,
		Path:        "/tmp/a.md",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		Status:      DocumentStatusIndexed,
	}); err != nil {
		t.Fatalf("UpsertDocument: %v", err)
	}
	oldVectorID := int64(1)
	active := Chunk{
		ID:             ChunkID(docID, 0, "active"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "active",
		ChunkText:      "active",
		VectorID:       &oldVectorID,
		EmbeddingModel: "old",
		Active:         true,
		CreatedAt:      now,
	}
	inactive := Chunk{
		ID:             ChunkID(docID, 1, "inactive"),
		DocumentID:     docID,
		ChunkIndex:     1,
		ContentHash:    "inactive",
		ChunkText:      "inactive",
		VectorID:       &oldVectorID,
		EmbeddingModel: "old",
		Active:         false,
		CreatedAt:      now,
	}
	if err := store.InsertChunk(ctx, active); err != nil {
		t.Fatalf("InsertChunk active: %v", err)
	}
	if err := store.InsertChunk(ctx, inactive); err != nil {
		t.Fatalf("InsertChunk inactive: %v", err)
	}
	if err := store.UpdateChunkVectorRefs(ctx, []ChunkVectorUpdate{
		{ChunkID: active.ID, VectorID: 42, EmbeddingModel: "new"},
	}); err != nil {
		t.Fatalf("UpdateChunkVectorRefs active: %v", err)
	}

	var activeVector int64
	var activeModel string
	if err := store.db.QueryRowContext(ctx, `select vector_id, embedding_model from chunks where id = ?`, active.ID).Scan(&activeVector, &activeModel); err != nil {
		t.Fatalf("read active chunk: %v", err)
	}
	if activeVector != 42 || activeModel != "new" {
		t.Fatalf("active vector/model = %d/%s, want 42/new", activeVector, activeModel)
	}

	err := store.UpdateChunkVectorRefs(ctx, []ChunkVectorUpdate{
		{ChunkID: inactive.ID, VectorID: 99, EmbeddingModel: "new"},
	})
	if err == nil || !strings.Contains(err.Error(), "affected 0 rows") {
		t.Fatalf("UpdateChunkVectorRefs inactive error = %v, want affected 0 rows", err)
	}
	err = store.UpdateChunkVectorRefs(ctx, []ChunkVectorUpdate{
		{ChunkID: "missing", VectorID: 100, EmbeddingModel: "new"},
	})
	if err == nil || !strings.Contains(err.Error(), "affected 0 rows") {
		t.Fatalf("UpdateChunkVectorRefs missing error = %v, want affected 0 rows", err)
	}
}

func TestUpdateChunkVectorRefsRollsBackWhenAnyUpdateFails(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/tmp/a.md")
	if err := store.UpsertDocument(ctx, Document{
		ID:          docID,
		Path:        "/tmp/a.md",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		Status:      DocumentStatusIndexed,
	}); err != nil {
		t.Fatalf("UpsertDocument: %v", err)
	}
	oldVectorID := int64(1)
	active := Chunk{
		ID:             ChunkID(docID, 0, "active"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "active",
		ChunkText:      "active",
		VectorID:       &oldVectorID,
		EmbeddingModel: "old",
		Active:         true,
		CreatedAt:      now,
	}
	inactive := Chunk{
		ID:             ChunkID(docID, 1, "inactive"),
		DocumentID:     docID,
		ChunkIndex:     1,
		ContentHash:    "inactive",
		ChunkText:      "inactive",
		VectorID:       &oldVectorID,
		EmbeddingModel: "old",
		Active:         false,
		CreatedAt:      now,
	}
	if err := store.InsertChunk(ctx, active); err != nil {
		t.Fatalf("InsertChunk active: %v", err)
	}
	if err := store.InsertChunk(ctx, inactive); err != nil {
		t.Fatalf("InsertChunk inactive: %v", err)
	}
	err := store.UpdateChunkVectorRefs(ctx, []ChunkVectorUpdate{
		{ChunkID: active.ID, VectorID: 42, EmbeddingModel: "new"},
		{ChunkID: inactive.ID, VectorID: 99, EmbeddingModel: "new"},
	})
	if err == nil || !strings.Contains(err.Error(), "affected 0 rows") {
		t.Fatalf("UpdateChunkVectorRefs error = %v, want inactive update failure", err)
	}

	var activeVector int64
	var activeModel string
	if err := store.db.QueryRowContext(ctx, `select vector_id, embedding_model from chunks where id = ?`, active.ID).Scan(&activeVector, &activeModel); err != nil {
		t.Fatalf("read active chunk: %v", err)
	}
	if activeVector != oldVectorID || activeModel != "old" {
		t.Fatalf("active vector/model = %d/%s, want rollback to old", activeVector, activeModel)
	}
}

func TestSearchLexicalChunksFindsChunkTextAndPath(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/repo/internal/rag/retrieve.go")
	if err := store.ReplaceDocumentChunks(ctx, Document{
		ID:          docID,
		Path:        "/repo/internal/rag/retrieve.go",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      DocumentStatusIndexed,
	}, []Chunk{{
		ID:             ChunkID(docID, 0, "chunk"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk",
		ChunkText:      "ErrPendingRunDetected is checked before compaction.",
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
	textHits, err := store.SearchLexicalChunks(ctx, "ErrPendingRunDetected", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks text: %v", err)
	}
	if len(textHits) != 1 {
		t.Fatalf("text hits = %+v, want one", textHits)
	}
	pathHits, err := store.SearchLexicalChunks(ctx, "internal/rag/retrieve.go", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks path: %v", err)
	}
	if len(pathHits) != 1 || pathHits[0].ChunkID != textHits[0].ChunkID {
		t.Fatalf("path hits = %+v, want same chunk", pathHits)
	}
}

func TestSearchLexicalChunksFindsHeadingPrefix(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/repo/docs/setup.md")
	chunkID := ChunkID(docID, 0, "chunk")
	if err := store.ReplaceDocumentChunks(ctx, Document{
		ID:          docID,
		Path:        "/repo/docs/setup.md",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      DocumentStatusIndexed,
	}, []Chunk{{
		ID:             chunkID,
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk",
		ChunkText:      "Set FALKENGO_EMBEDDING_MODEL_API_KEY before running.",
		IndexedText:    "Document: docs/setup.md\nSection: Project > Installation > Environment\nChunk:\nSet FALKENGO_EMBEDDING_MODEL_API_KEY before running.",
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
	hits, err := store.SearchLexicalChunks(ctx, "Environment", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks: %v", err)
	}
	if len(hits) != 1 || hits[0].ChunkID != chunkID {
		t.Fatalf("hits = %+v, want heading-prefix chunk", hits)
	}
}

func TestSearchLexicalChunksFindsSymbolPrefix(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/repo/internal/rag/retrieve.go")
	chunkID := ChunkID(docID, 0, "chunk")
	if err := store.ReplaceDocumentChunks(ctx, Document{
		ID:          docID,
		Path:        "/repo/internal/rag/retrieve.go",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      DocumentStatusIndexed,
	}, []Chunk{{
		ID:             chunkID,
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk",
		ChunkText:      "return chunks, nil",
		IndexedText:    "File: internal/rag/retrieve.go\nLanguage: go\nSymbol: function Retrieve\nChunk:\nreturn chunks, nil",
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
	hits, err := store.SearchLexicalChunks(ctx, "Retrieve", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks: %v", err)
	}
	if len(hits) != 1 || hits[0].ChunkID != chunkID {
		t.Fatalf("hits = %+v, want symbol-prefix chunk", hits)
	}
}

func TestSearchLexicalChunksExcludesInactiveAndDeleted(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	activeDocID := DocumentID("/repo/active.md")
	deletedDocID := DocumentID("/repo/deleted.md")
	inactiveDocID := DocumentID("/repo/inactive.md")
	docs := []Document{
		{ID: activeDocID, Path: "/repo/active.md", ContentHash: "a", SizeBytes: 1, ModifiedAt: now, IndexedAt: &now, Status: DocumentStatusIndexed},
		{ID: deletedDocID, Path: "/repo/deleted.md", ContentHash: "d", SizeBytes: 1, ModifiedAt: now, Status: DocumentStatusDeleted},
		{ID: inactiveDocID, Path: "/repo/inactive.md", ContentHash: "i", SizeBytes: 1, ModifiedAt: now, IndexedAt: &now, Status: DocumentStatusIndexed},
	}
	for _, doc := range docs {
		if err := store.UpsertDocument(ctx, doc); err != nil {
			t.Fatalf("UpsertDocument: %v", err)
		}
	}
	chunks := []Chunk{
		{ID: ChunkID(activeDocID, 0, "active"), DocumentID: activeDocID, ChunkIndex: 0, ContentHash: "active", ChunkText: "sharedtoken active", EmbeddingModel: "model", Active: true, CreatedAt: now},
		{ID: ChunkID(deletedDocID, 0, "deleted"), DocumentID: deletedDocID, ChunkIndex: 0, ContentHash: "deleted", ChunkText: "sharedtoken deleted", EmbeddingModel: "model", Active: true, CreatedAt: now},
		{ID: ChunkID(inactiveDocID, 0, "inactive"), DocumentID: inactiveDocID, ChunkIndex: 0, ContentHash: "inactive", ChunkText: "sharedtoken inactive", EmbeddingModel: "model", Active: false, CreatedAt: now},
	}
	for _, chunk := range chunks {
		if err := store.InsertChunk(ctx, chunk); err != nil {
			t.Fatalf("InsertChunk: %v", err)
		}
	}
	hits, err := store.SearchLexicalChunks(ctx, "sharedtoken", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks: %v", err)
	}
	if len(hits) != 1 || hits[0].ChunkID != chunks[0].ID {
		t.Fatalf("hits = %+v, want only active indexed chunk", hits)
	}
}

func TestRebuildLexicalIndexBackfillsActiveIndexedChunks(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/repo/readme.md")
	if err := store.UpsertDocument(ctx, Document{
		ID:          docID,
		Path:        "/repo/readme.md",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      DocumentStatusIndexed,
	}); err != nil {
		t.Fatalf("UpsertDocument: %v", err)
	}
	chunk := Chunk{ID: ChunkID(docID, 0, "chunk"), DocumentID: docID, ChunkIndex: 0, ContentHash: "chunk", ChunkText: "backfilltoken", EmbeddingModel: "model", Active: true, CreatedAt: now}
	if _, err := store.db.ExecContext(ctx, `
insert into chunks(id, document_id, chunk_index, content_hash, chunk_text, embedding_model, active, created_at)
values (?, ?, ?, ?, ?, ?, 1, ?)
`, chunk.ID, chunk.DocumentID, chunk.ChunkIndex, chunk.ContentHash, chunk.ChunkText, chunk.EmbeddingModel, formatTime(chunk.CreatedAt)); err != nil {
		t.Fatalf("raw insert chunk: %v", err)
	}
	if err := store.RebuildLexicalIndex(ctx); err != nil {
		t.Fatalf("RebuildLexicalIndex: %v", err)
	}
	hits, err := store.SearchLexicalChunks(ctx, "backfilltoken", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks: %v", err)
	}
	if len(hits) != 1 || hits[0].ChunkID != chunk.ID {
		t.Fatalf("hits = %+v, want backfilled chunk", hits)
	}
}

func TestEnsureLexicalIndexRebuildsOnlyWhenEmpty(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/repo/readme.md")
	if err := store.ReplaceDocumentChunks(ctx, Document{
		ID:          docID,
		Path:        "/repo/readme.md",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      DocumentStatusIndexed,
	}, []Chunk{{
		ID:             ChunkID(docID, 0, "chunk"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk",
		ChunkText:      "ensuretoken",
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
	rebuilt, err := store.EnsureLexicalIndex(ctx)
	if err != nil {
		t.Fatalf("EnsureLexicalIndex populated: %v", err)
	}
	if rebuilt {
		t.Fatal("EnsureLexicalIndex rebuilt populated index, want no rebuild")
	}
	if _, err := store.db.ExecContext(ctx, `delete from chunk_fts`); err != nil {
		t.Fatalf("clear fts: %v", err)
	}
	rebuilt, err = store.EnsureLexicalIndex(ctx)
	if err != nil {
		t.Fatalf("EnsureLexicalIndex empty: %v", err)
	}
	if !rebuilt {
		t.Fatal("EnsureLexicalIndex rebuilt=false, want rebuild for empty index with active chunks")
	}
	hits, err := store.SearchLexicalChunks(ctx, "ensuretoken", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks: %v", err)
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want rebuilt lexical hit", hits)
	}
}

func TestReplaceDocumentChunksAndMarkChunksInactiveUpdateLexicalIndex(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/repo/readme.md")
	doc := Document{ID: docID, Path: "/repo/readme.md", ContentHash: "old", SizeBytes: 1, ModifiedAt: now, IndexedAt: &now, Status: DocumentStatusIndexed}
	oldChunk := Chunk{ID: ChunkID(docID, 0, "old"), DocumentID: docID, ChunkIndex: 0, ContentHash: "old", ChunkText: "oldtoken", EmbeddingModel: "model", Active: true, CreatedAt: now}
	if err := store.ReplaceDocumentChunks(ctx, doc, []Chunk{oldChunk}); err != nil {
		t.Fatalf("ReplaceDocumentChunks old: %v", err)
	}
	newChunk := oldChunk
	newChunk.ID = ChunkID(docID, 0, "new")
	newChunk.ContentHash = "new"
	newChunk.ChunkText = "newtoken"
	if err := store.ReplaceDocumentChunks(ctx, doc, []Chunk{newChunk}); err != nil {
		t.Fatalf("ReplaceDocumentChunks new: %v", err)
	}
	oldHits, err := store.SearchLexicalChunks(ctx, "oldtoken", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks old: %v", err)
	}
	if len(oldHits) != 0 {
		t.Fatalf("old hits = %+v, want removed", oldHits)
	}
	newHits, err := store.SearchLexicalChunks(ctx, "newtoken", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks new: %v", err)
	}
	if len(newHits) != 1 {
		t.Fatalf("new hits = %+v, want new chunk", newHits)
	}
	if err := store.MarkChunksInactive(ctx, docID); err != nil {
		t.Fatalf("MarkChunksInactive: %v", err)
	}
	newHits, err = store.SearchLexicalChunks(ctx, "newtoken", 10)
	if err != nil {
		t.Fatalf("SearchLexicalChunks after inactive: %v", err)
	}
	if len(newHits) != 0 {
		t.Fatalf("hits after inactive = %+v, want none", newHits)
	}
}

func TestBuildFTSQueryHandlesCodeSymbolsAndFlags(t *testing.T) {
	for _, input := range []string{
		"ErrPendingRunDetected",
		"--state-dir",
		"FALKENGO_EMBEDDING_MODEL",
		"internal/rag/retrieve.go",
		"what does c.active = 1 mean?",
	} {
		if got := BuildFTSQuery(input); got == "" {
			t.Fatalf("BuildFTSQuery(%q) returned empty query", input)
		}
	}
}

func TestSearchLexicalChunksHandlesCodeQueries(t *testing.T) {
	store := openTestStore(t)
	ctx := context.Background()
	now := time.Now().UTC()
	docID := DocumentID("/repo/internal/rag/retrieve.go")
	if err := store.ReplaceDocumentChunks(ctx, Document{
		ID:          docID,
		Path:        "/repo/internal/rag/retrieve.go",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      DocumentStatusIndexed,
	}, []Chunk{{
		ID:             ChunkID(docID, 0, "chunk"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk",
		ChunkText:      "ErrPendingRunDetected --state-dir FALKENGO_EMBEDDING_MODEL c.active = 1",
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      now,
	}}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
	for _, input := range []string{
		"ErrPendingRunDetected",
		"--state-dir",
		"FALKENGO_EMBEDDING_MODEL",
		"internal/rag/retrieve.go",
		"what does c.active = 1 mean?",
	} {
		if _, err := store.SearchLexicalChunks(ctx, input, 10); err != nil {
			t.Fatalf("SearchLexicalChunks(%q): %v", input, err)
		}
	}
}

func TestForeignKeysAreEnforced(t *testing.T) {
	store := openTestStore(t)
	err := store.InsertChunk(context.Background(), Chunk{
		ID:             "chunk",
		DocumentID:     "missing-doc",
		ChunkIndex:     0,
		ContentHash:    "hash",
		ChunkText:      "text",
		EmbeddingModel: "model",
		Active:         true,
		CreatedAt:      time.Now().UTC(),
	})
	if err == nil {
		t.Fatal("InsertChunk succeeded for missing document, want foreign-key error")
	}
}

func openTestStore(t *testing.T) *SQLiteStore {
	t.Helper()
	store, err := Open(filepath.Join(t.TempDir(), "manifest.sqlite"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if err := store.Init(context.Background()); err != nil {
		t.Fatalf("Init: %v", err)
	}
	return store
}

func sameStringSlice(got []string, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestEnsureColumnValidation(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "manifest.sqlite")
	store, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer store.Close()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("Init: %v", err)
	}

	tests := []struct {
		name       string
		table      string
		column     string
		columnType string
		wantErr    bool
	}{
		{
			name:       "valid inputs",
			table:      "documents",
			column:     "new_col",
			columnType: "text not null default ''",
			wantErr:    false,
		},
		{
			name:       "invalid table",
			table:      "documents; drop table documents;--",
			column:     "new_col",
			columnType: "text",
			wantErr:    true,
		},
		{
			name:       "invalid column",
			table:      "documents",
			column:     "new_col; drop table documents;--",
			columnType: "text",
			wantErr:    true,
		},
		{
			name:       "invalid column type",
			table:      "documents",
			column:     "new_col",
			columnType: "text; drop table documents;--",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ensureColumn(ctx, store.db, tt.table, tt.column, tt.columnType)
			if (err != nil) != tt.wantErr {
				t.Errorf("ensureColumn() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
