package cli

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/spf13/cobra"
)

func TestVersionCommand(t *testing.T) {
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"version"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "falkengo") {
		t.Fatalf("output = %q", out.String())
	}
}

func TestHelpCommand(t *testing.T) {
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--help"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "ingest") || !strings.Contains(out.String(), "query") || !strings.Contains(out.String(), "compact") || !strings.Contains(out.String(), "eval") {
		t.Fatalf("help output = %q", out.String())
	}
}

func TestIngestHelpDescribesExtensionsAsRestriction(t *testing.T) {
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"ingest", "--help"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "restrict indexing to comma-separated file extensions") {
		t.Fatalf("ingest help output = %q, want --extensions restriction help", out.String())
	}
	if !strings.Contains(out.String(), "exclude comma-separated file extensions") {
		t.Fatalf("ingest help output = %q, want --exclude-extensions help", out.String())
	}
	if !strings.Contains(out.String(), "exclude comma-separated directory names") {
		t.Fatalf("ingest help output = %q, want --exclude-dirs help", out.String())
	}
}

func TestCommandContextUsesDefaultTimeout(t *testing.T) {
	cmd := NewRootCommand()
	ctx, cancel := commandContext(cmd, &options{timeout: config.DefaultTimeout})
	defer cancel()
	if _, ok := ctx.Deadline(); !ok {
		t.Fatal("commandContext has no deadline, want default timeout deadline")
	}
}

func TestLongRunningCommandContextHasNoDefaultDeadline(t *testing.T) {
	for _, tt := range []struct {
		name string
		cmd  func(*options) any
	}{
		{name: "ingest", cmd: func(opts *options) any { return newIngestCommand(opts) }},
		{name: "compact", cmd: func(opts *options) any { return newCompactCommand(opts) }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			opts := &options{timeout: config.DefaultTimeout}
			cmd := tt.cmd(opts).(*cobra.Command)
			ctx, cancel := longRunningCommandContext(cmd, opts)
			defer cancel()
			if _, ok := ctx.Deadline(); ok {
				t.Fatal("longRunningCommandContext has deadline, want none without explicit --timeout")
			}
		})
	}
}

func TestLongRunningCommandContextHonorsExplicitTimeout(t *testing.T) {
	opts := &options{timeout: time.Millisecond}
	cmd := NewRootCommand()
	flag := cmd.PersistentFlags().Lookup("timeout")
	if flag == nil {
		t.Fatal("timeout flag not found")
	}
	if err := flag.Value.Set(opts.timeout.String()); err != nil {
		t.Fatalf("set timeout flag: %v", err)
	}
	flag.Changed = true

	ctx, cancel := longRunningCommandContext(cmd, opts)
	defer cancel()
	if _, ok := ctx.Deadline(); !ok {
		t.Fatal("longRunningCommandContext has no deadline, want explicit timeout deadline")
	}
}

func TestStatusBeforeIngest(t *testing.T) {
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", filepath.Join(t.TempDir(), ".falkengo"), "status"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "No index found") {
		t.Fatalf("status output = %q", out.String())
	}
}

func TestResetConfirmationNoDoesNotDelete(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetIn(strings.NewReader("n\n"))
	cmd.SetArgs([]string{"--state-dir", state, "reset"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "reset cancelled") {
		t.Fatalf("reset output = %q", out.String())
	}
}

func TestResetYesDeletesStateDir(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	if err := os.MkdirAll(state, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(state, "manifest.sqlite"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "reset", "--yes"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if _, err := os.Stat(state); !os.IsNotExist(err) {
		t.Fatalf("state dir still exists, stat err = %v", err)
	}
}

func TestResetRefusesUnsafeStateDirWithoutForce(t *testing.T) {
	state := filepath.Join(t.TempDir(), "custom-state")
	if err := os.MkdirAll(state, 0o755); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "reset", "--yes"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "basename is not .falkengo") {
		t.Fatalf("Execute error = %v, want unsafe state dir refusal", err)
	}
	if _, err := os.Stat(state); err != nil {
		t.Fatalf("state dir was removed or inaccessible: %v", err)
	}
}

func TestIngestDryRunDoesNotCreateStateDirOrNeedAPIKey(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "README"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	state := filepath.Join(root, ".falkengo")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ingest", root, "--dry-run"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if _, err := os.Stat(state); !os.IsNotExist(err) {
		t.Fatalf("state dir stat = %v, want not created", err)
	}
	if !strings.Contains(out.String(), "new files: 1") {
		t.Fatalf("dry-run output = %q", out.String())
	}
}

func TestIngestDryRunRespectsExcludes(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "keep.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "skip.md"), []byte("# Skip\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "fixtures"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "fixtures", "fixture.go"), []byte("package fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	state := filepath.Join(root, ".falkengo")
	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{
		"--state-dir", state,
		"ingest", root,
		"--dry-run",
		"--extensions", "go,md",
		"--exclude-extensions", "md",
		"--exclude-dirs", "fixtures",
	})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, "scanned: 1") || !strings.Contains(output, "new files: 1") {
		t.Fatalf("dry-run output = %q, want only non-excluded file", output)
	}
	if _, err := os.Stat(state); !os.IsNotExist(err) {
		t.Fatalf("state dir stat = %v, want not created", err)
	}
}

func TestIngestDryRunWorksWithPreMigrationManifestWithoutMigrating(t *testing.T) {
	root := t.TempDir()
	readme := filepath.Join(root, "README.md")
	if err := os.WriteFile(readme, []byte("# Hello\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	state := filepath.Join(root, ".falkengo")
	if err := os.MkdirAll(state, 0o755); err != nil {
		t.Fatal(err)
	}
	manifestPath := filepath.Join(state, "manifest.sqlite")
	db, err := sql.Open("sqlite", manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	if _, err := db.ExecContext(context.Background(), `
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
insert into documents(id, path, content_hash, size_bytes, modified_at, indexed_at, status, error)
values (?, ?, ?, ?, ?, ?, ?, null);
`, manifest.DocumentID(readme), readme, "oldhash", int64(8), now, now, manifest.DocumentStatusIndexed); err != nil {
		db.Close()
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "ingest", root, "--dry-run"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "changed files: 1") {
		t.Fatalf("dry-run output = %q, want changed file summary", out.String())
	}

	db, err = sql.Open("sqlite", manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if hasColumn(t, db, "documents", "chunker") {
		t.Fatal("dry-run migrated documents.chunker column, want zero manifest writes")
	}
}

func TestIngestRejectsInvalidChunker(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"ingest", root, "--dry-run", "--chunker", "bogus"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "--chunker must be auto, fixed, markdown, text, or code") {
		t.Fatalf("Execute error = %v, want invalid chunker error", err)
	}
}

func TestQueryChecksIndexBeforeEmbedderEnv(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths.ManifestPath, []byte("not enough"), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "query", "hello"})
	err = cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "No index found") {
		t.Fatalf("Execute error = %v, want local index error before API key error", err)
	}
}

func TestStatusAfterIndexPrintsCounts(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	store, err := manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	if err := store.UpsertDocument(context.Background(), manifest.Document{
		ID:          manifest.DocumentID("README.md"),
		Path:        "README.md",
		ContentHash: "hash",
		SizeBytes:   1,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      manifest.DocumentStatusIndexed,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.UpsertDocument(context.Background(), manifest.Document{
		ID:          manifest.DocumentID("old.md"),
		Path:        "old.md",
		ContentHash: "old",
		SizeBytes:   1,
		ModifiedAt:  now,
		Status:      manifest.DocumentStatusDeleted,
		DeletedAt:   &now,
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "status"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "indexed: 1") || !strings.Contains(out.String(), "deleted: 1") {
		t.Fatalf("status output = %q", out.String())
	}
}

func TestCompactDryRunPrintsSummaryAndUsesStateDir(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	store, err := manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	docID := manifest.DocumentID("README.md")
	vectorID := int64(1)
	doc := manifest.Document{
		ID:          docID,
		Path:        "README.md",
		ContentHash: "hash",
		SizeBytes:   10,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      manifest.DocumentStatusIndexed,
	}
	if err := store.ReplaceDocumentChunks(context.Background(), doc, []manifest.Chunk{
		{
			ID:             manifest.ChunkID(docID, 0, "active"),
			DocumentID:     docID,
			ChunkIndex:     0,
			ContentHash:    "active",
			ChunkText:      "active",
			VectorID:       &vectorID,
			EmbeddingModel: "old",
			Active:         true,
			CreatedAt:      now,
		},
		{
			ID:             manifest.ChunkID(docID, 1, "inactive"),
			DocumentID:     docID,
			ChunkIndex:     1,
			ContentHash:    "inactive",
			ChunkText:      "inactive",
			VectorID:       &vectorID,
			EmbeddingModel: "old",
			Active:         false,
			CreatedAt:      now,
		},
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "compact", "--dry-run"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	output := out.String()
	if !strings.Contains(output, "compact dry run") || !strings.Contains(output, "active chunks: 1") || !strings.Contains(output, "inactive chunks: 1") {
		t.Fatalf("compact dry-run output = %q", output)
	}
	if !strings.Contains(output, paths.VecgoPath) || !strings.Contains(output, paths.StateDir) {
		t.Fatalf("compact dry-run output does not use resolved state paths: %q", output)
	}
	if _, err := os.Stat(paths.VecgoPath); !os.IsNotExist(err) {
		t.Fatalf("vecgo path stat = %v, want not created by dry-run", err)
	}
}

func TestCompactDryRunWarnsOnPendingRun(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	_, store := stagedPendingRunForCLITest(t, state)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "compact", "--dry-run"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "warning: pending ingest run detected") {
		t.Fatalf("compact dry-run output = %q, want pending-run warning", out.String())
	}
}

func TestCompactBeforeIngestReturnsNoIndex(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "compact"})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "No index found") {
		t.Fatalf("Execute error = %v, want no index", err)
	}
}

func TestCompactFailsWhenWriteLockHeld(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	store, err := manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	lock, err := config.AcquireWriteLock(paths)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Release()

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "compact"})
	err = cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "write lock is already held") {
		t.Fatalf("Execute error = %v, want write lock error", err)
	}
}

func TestRepairActivatesPendingRun(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	paths, store := stagedPendingRunForCLITest(t, state)
	_ = paths
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "repair"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "activated pending ingest run") {
		t.Fatalf("repair output = %q", out.String())
	}

	store, err := manifest.Open(filepath.Join(state, "manifest.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	runs, err := store.ListPendingRuns(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 0 {
		t.Fatalf("pending runs = %+v, want none", runs)
	}
	doc, err := store.GetDocumentByPath(context.Background(), "README.md")
	if err != nil {
		t.Fatal(err)
	}
	chunks, err := store.GetActiveChunksByIDs(context.Background(), []string{manifest.ChunkID(doc.ID, 0, "chunk-hash")})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("active chunks = %+v, want staged chunk", chunks)
	}
}

func TestRepairDiscardsPendingRun(t *testing.T) {
	state := filepath.Join(t.TempDir(), ".falkengo")
	_, store := stagedPendingRunForCLITest(t, state)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetArgs([]string{"--state-dir", state, "repair", "--discard-pending"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if !strings.Contains(out.String(), "discarded pending ingest run") {
		t.Fatalf("repair output = %q", out.String())
	}

	store, err := manifest.Open(filepath.Join(state, "manifest.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	runs, err := store.ListPendingRuns(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(runs) != 0 {
		t.Fatalf("pending runs = %+v, want none", runs)
	}
	if _, err := store.GetDocumentByPath(context.Background(), "README.md"); !errors.Is(err, manifest.ErrNotFound) {
		t.Fatalf("GetDocumentByPath error = %v, want not found", err)
	}
}

func TestIngestReportsPendingRun(t *testing.T) {
	root := t.TempDir()
	state := filepath.Join(root, ".falkengo")
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("hello"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, store := stagedPendingRunForCLITest(t, state)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := NewRootCommand()
	cmd.SetArgs([]string{"--state-dir", state, "ingest", root})
	err := cmd.Execute()
	if err == nil || !strings.Contains(err.Error(), "Pending ingest run detected") {
		t.Fatalf("Execute error = %v, want pending-run message", err)
	}
}

func stagedPendingRunForCLITest(t *testing.T, state string) (config.Paths, *manifest.SQLiteStore) {
	t.Helper()
	paths, err := config.ResolvePaths(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := config.EnsureStateDirs(paths); err != nil {
		t.Fatal(err)
	}
	store, err := manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Init(context.Background()); err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC()
	docID := manifest.DocumentID("README.md")
	vectorID := int64(1)
	if err := store.StagePendingDocument(context.Background(), "run-1", manifest.Document{
		ID:          docID,
		Path:        "README.md",
		ContentHash: "doc-hash",
		SizeBytes:   5,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      manifest.DocumentStatusIndexed,
	}, []manifest.Chunk{{
		ID:             manifest.ChunkID(docID, 0, "chunk-hash"),
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "chunk-hash",
		ChunkText:      "hello",
		StartLine:      1,
		EndLine:        1,
		VectorID:       &vectorID,
		EmbeddingModel: "fake",
		Active:         true,
		CreatedAt:      now,
	}}); err != nil {
		t.Fatal(err)
	}
	return paths, store
}

func hasColumn(t *testing.T, db *sql.DB, table string, column string) bool {
	t.Helper()
	rows, err := db.QueryContext(context.Background(), `pragma table_info(`+table+`)`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name string
		var columnType string
		var notNull int
		var defaultValue sql.NullString
		var primaryKey int
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			t.Fatal(err)
		}
		if name == column {
			return true
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	return false
}
