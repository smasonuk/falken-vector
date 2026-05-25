package ingest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/vectorstore"
)

var ErrPendingRunDetected = errors.New("Pending ingest run detected. Run `falkengo repair`.")

type Options struct {
	Root         string
	Paths        config.Paths
	Extensions   []string
	ChunkSize    int
	ChunkOverlap int
	ChunkerMode  ChunkerMode
	DryRun       bool
	SyncSource   bool
	Verbose      bool
	Out          io.Writer
	Embedder     llm.Embedder
	OpenVector   func(context.Context, string, int) (vectorstore.Store, error)
	Now          func() time.Time
	Progress     func(ProgressEvent)
}

type Summary struct {
	Scanned        int
	NewFiles       int
	ChangedFiles   int
	UnchangedFiles int
	DeletedFiles   int
	FailedFiles    int
	ChunksEmbedded int
	StateDir       string
}

type ProgressEvent struct {
	Directory   string
	CurrentPath string
	CurrentFile int
	TotalFiles  int
	Action      string

	Scanned        int
	NewFiles       int
	ChangedFiles   int
	UnchangedFiles int
	DeletedFiles   int
	FailedFiles    int
	ChunksEmbedded int
}

type chunkReplacer interface {
	ReplaceDocumentChunks(context.Context, manifest.Document, []manifest.Chunk) error
}

type pendingIndexer interface {
	manifest.PendingRunStore
	StagePendingDocument(context.Context, string, manifest.Document, []manifest.Chunk) error
}

func Run(ctx context.Context, store manifest.Store, opts Options) (Summary, error) {
	if opts.ChunkSize == 0 {
		opts.ChunkSize = 1200
	}
	if opts.ChunkerMode == "" {
		opts.ChunkerMode = ChunkerModeAuto
	}
	if opts.Out == nil {
		opts.Out = io.Discard
	}
	if opts.Now == nil {
		opts.Now = func() time.Time { return time.Now().UTC() }
	}
	if opts.OpenVector == nil {
		opts.OpenVector = vectorstore.Open
	}
	if !opts.DryRun && opts.Embedder == nil {
		return Summary{}, fmt.Errorf("embedder is required")
	}
	sourceRoot, err := resolveRoot(opts.Root)
	if err != nil {
		return Summary{}, fmt.Errorf("resolve ingest root: %w", err)
	}
	if !opts.DryRun {
		if pendingStore, ok := store.(manifest.PendingRunStore); ok {
			runs, err := pendingStore.ListPendingRuns(ctx)
			if err != nil {
				if isContextStop(ctx, err) {
					return Summary{}, ingestStoppedError(ctx, err)
				}
				return Summary{}, fmt.Errorf("list pending ingest runs: %w", err)
			}
			if len(runs) != 0 {
				return Summary{}, ErrPendingRunDetected
			}
		}
	}

	progressf(opts, "scanning %s\n", sourceRoot)
	files, err := FindCandidateFiles(ctx, WalkOptions{
		Root:       sourceRoot,
		StateDir:   opts.Paths.StateDir,
		Extensions: opts.Extensions,
	})
	if err != nil {
		if isContextStop(ctx, err) {
			return Summary{}, ingestStoppedError(ctx, err)
		}
		return Summary{}, fmt.Errorf("walk candidate files: %w", err)
	}
	summary := Summary{Scanned: len(files), StateDir: opts.Paths.StateDir}
	progressf(opts, "found %d candidate files\n", len(files))
	emitProgress(opts, ProgressEvent{
		Directory:  sourceRoot,
		Action:     "scanned",
		Scanned:    len(files),
		TotalFiles: len(files),
	})
	discoveredPaths := make(map[string]struct{}, len(files))
	var vectorDB vectorstore.Store
	pending := make([]indexedDocument, 0)
	pendingStore, usePendingStore := store.(pendingIndexer)
	runID := ""
	if !opts.DryRun && usePendingStore {
		runID, err = newRunID()
		if err != nil {
			return summary, fmt.Errorf("create index run id: %w", err)
		}
	}
	defer func() {
		if vectorDB != nil {
			_ = vectorDB.Close()
		}
	}()

	for i, candidate := range files {
		if err := ctx.Err(); err != nil {
			clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
			return summary, ingestStoppedError(ctx, err)
		}
		discoveredPaths[candidate.Path] = struct{}{}
		effectiveChunker := EffectiveChunkerMode(candidate.Path, opts.ChunkerMode)
		decision, contentHash, err := DecideFileWithConfig(ctx, store, candidate, ChunkConfig{
			Chunker:          string(effectiveChunker),
			ChunkSize:        opts.ChunkSize,
			ChunkOverlap:     opts.ChunkOverlap,
			IndexTextVersion: CurrentIndexTextVersion,
		})
		if err != nil {
			if isContextStop(ctx, err) {
				clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
				return summary, ingestStoppedError(ctx, err)
			}
			summary.FailedFiles++
			if !opts.DryRun {
				_ = store.MarkDocumentError(ctx, candidate.Path, err.Error())
			}
			verbosef(opts, "failed to inspect %s: %v\n", candidate.Path, err)
			continue
		}
		switch decision {
		case FileDecisionUnchanged:
			summary.UnchangedFiles++
			if opts.DryRun {
				verbosef(opts, "would skip unchanged file: %s\n", candidate.Path)
			} else {
				verbosef(opts, "skipping unchanged %s\n", candidate.Path)
			}
			continue
		case FileDecisionNew:
			summary.NewFiles++
			if opts.DryRun {
				verbosef(opts, "would index new file: %s\n", candidate.Path)
			}
		case FileDecisionChanged:
			summary.ChangedFiles++
			if opts.DryRun {
				verbosef(opts, "would re-index changed file: %s\n", candidate.Path)
			}
		}

		progressFileDecision(opts, sourceRoot, candidate.Path, decision, i+1, len(files))
		indexed, err := indexFile(ctx, &vectorDB, candidate, sourceRoot, contentHash, effectiveChunker, opts)
		if err != nil {
			if isContextStop(ctx, err) {
				clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
				return summary, ingestStoppedError(ctx, err)
			}
			summary.FailedFiles++
			if !opts.DryRun {
				_ = markCandidateError(ctx, store, candidate, contentHash, opts.Now(), err)
			}
			verbosef(opts, "failed to index %s: %v\n", candidate.Path, err)
			continue
		}
		if !opts.DryRun && usePendingStore {
			if err := pendingStore.StagePendingDocument(ctx, runID, indexed.Document, indexed.Chunks); err != nil {
				if isContextStop(ctx, err) {
					clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
					return summary, ingestStoppedError(ctx, err)
				}
				_ = pendingStore.ClearPendingRun(context.Background(), runID)
				return summary, fmt.Errorf("stage manifest rows for %s: %w", indexed.Document.Path, err)
			}
		} else {
			pending = append(pending, indexed)
		}
		summary.ChunksEmbedded += len(indexed.Chunks)
	}

	if err := ctx.Err(); err != nil {
		clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
		return summary, ingestStoppedError(ctx, err)
	}
	deletedIDs, err := deletedDocumentsForSync(ctx, store, opts.SyncSource, sourceRoot, discoveredPaths, opts)
	if err != nil {
		if isContextStop(ctx, err) {
			clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
			return summary, ingestStoppedError(ctx, err)
		}
		return summary, err
	}
	summary.DeletedFiles = len(deletedIDs)

	if vectorDB != nil && !opts.DryRun {
		if err := ctx.Err(); err != nil {
			clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
			return summary, ingestStoppedError(ctx, err)
		}
		progressf(opts, "committing vector database\n")
		if err := vectorDB.Commit(ctx); err != nil {
			if isContextStop(ctx, err) {
				clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
				return summary, ingestStoppedError(ctx, err)
			}
			if usePendingStore {
				_ = pendingStore.ClearPendingRun(context.Background(), runID)
			}
			return summary, fmt.Errorf("commit vector database: %w", err)
		}
	}
	if !opts.DryRun {
		if err := ctx.Err(); err != nil {
			clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
			return summary, ingestStoppedError(ctx, err)
		}
		progressf(opts, "updating manifest\n")
		if usePendingStore {
			if err := pendingStore.ActivatePendingRun(ctx, runID); err != nil {
				if isContextStop(ctx, err) {
					clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
					return summary, ingestStoppedError(ctx, err)
				}
				return summary, fmt.Errorf("activate staged manifest rows: %w", err)
			}
		} else {
			for _, indexed := range pending {
				if err := writeIndexedDocument(ctx, store, indexed); err != nil {
					if isContextStop(ctx, err) {
						clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
						return summary, ingestStoppedError(ctx, err)
					}
					return summary, err
				}
			}
		}
		if len(deletedIDs) != 0 {
			if err := store.MarkDocumentsDeleted(ctx, deletedIDs, opts.Now()); err != nil {
				if isContextStop(ctx, err) {
					clearPendingRunAfterStop(pendingStore, usePendingStore, runID)
					return summary, ingestStoppedError(ctx, err)
				}
				return summary, fmt.Errorf("mark deleted source documents: %w", err)
			}
		}
	}
	processedCandidates := summary.Scanned - summary.UnchangedFiles
	if processedCandidates > 0 && summary.FailedFiles == processedCandidates {
		return summary, fmt.Errorf("all files failed to index")
	}
	return summary, nil
}

type indexedDocument struct {
	Document manifest.Document
	Chunks   []manifest.Chunk
}

func indexFile(ctx context.Context, vectorDB *vectorstore.Store, candidate CandidateFile, sourceRoot string, contentHash string, effectiveChunker ChunkerMode, opts Options) (indexedDocument, error) {
	body, err := os.ReadFile(candidate.Path)
	if err != nil {
		return indexedDocument{}, fmt.Errorf("read file: %w", err)
	}
	chunks, err := ChunkText(string(body), ChunkOptions{ChunkSize: opts.ChunkSize, ChunkOverlap: opts.ChunkOverlap, Mode: effectiveChunker, Path: candidate.Path})
	if err != nil {
		return indexedDocument{}, fmt.Errorf("chunk file: %w", err)
	}
	verbosef(opts, "indexing %s with %s chunker (%d chunks)\n", candidate.Path, effectiveChunker, len(chunks))
	if opts.DryRun {
		progressf(opts, "would chunk %s into %d chunks with %s chunker\n", displayPath(sourceRoot, candidate.Path), len(chunks), effectiveChunker)
		return indexedDocument{}, nil
	}
	progressf(opts, "embedding %d chunks from %s with %s chunker\n", len(chunks), displayPath(sourceRoot, candidate.Path), effectiveChunker)

	docID := manifest.DocumentID(candidate.Path)
	now := opts.Now()
	indexedAt := now
	doc := manifest.Document{
		ID:               docID,
		Path:             candidate.Path,
		ContentHash:      contentHash,
		SizeBytes:        candidate.SizeBytes,
		ModifiedAt:       candidate.ModifiedAt.UTC(),
		IndexedAt:        &indexedAt,
		SourceRoot:       sourceRoot,
		Chunker:          string(effectiveChunker),
		ChunkSize:        opts.ChunkSize,
		ChunkOverlap:     opts.ChunkOverlap,
		IndexTextVersion: CurrentIndexTextVersion,
		Status:           manifest.DocumentStatusIndexed,
	}
	rows := make([]manifest.Chunk, 0, len(chunks))
	for _, chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return indexedDocument{}, err
		}
		indexedText := BuildIndexedChunkText(candidate.Path, chunk)
		embedding, err := opts.Embedder.EmbedText(ctx, indexedText)
		if err != nil {
			return indexedDocument{}, fmt.Errorf("embed chunk %d for %s: %w", chunk.Index, candidate.Path, err)
		}
		if len(embedding.Vector) == 0 {
			return indexedDocument{}, fmt.Errorf("embed chunk %d for %s: empty vector", chunk.Index, candidate.Path)
		}
		if *vectorDB == nil {
			store, err := opts.OpenVector(ctx, opts.Paths.VecgoPath, len(embedding.Vector))
			if err != nil {
				return indexedDocument{}, fmt.Errorf("open vector database: %w", err)
			}
			*vectorDB = store
		}
		chunkID := manifest.ChunkID(docID, chunk.Index, chunk.Hash)
		vectorID, err := (*vectorDB).Insert(ctx, embedding.Vector, chunkID)
		if err != nil {
			return indexedDocument{}, fmt.Errorf("insert vector for chunk %d in %s: %w", chunk.Index, candidate.Path, err)
		}
		progressChunkEmbedded(opts, displayPath(sourceRoot, candidate.Path), chunk.Index+1, len(chunks))
		rows = append(rows, manifest.Chunk{
			ID:             chunkID,
			DocumentID:     docID,
			ChunkIndex:     chunk.Index,
			ContentHash:    chunk.Hash,
			ChunkText:      chunk.Text,
			IndexedText:    indexedText,
			StartLine:      chunk.StartLine,
			EndLine:        chunk.EndLine,
			VectorID:       &vectorID,
			EmbeddingModel: embedding.Model,
			Active:         true,
			CreatedAt:      now,
			Chunker:        chunk.Chunker,
			Language:       chunk.Language,
			HeadingPath:    chunk.HeadingPath,
			SymbolName:     chunk.SymbolName,
			SymbolKind:     chunk.SymbolKind,
		})
	}
	return indexedDocument{Document: doc, Chunks: rows}, nil
}

func writeIndexedDocument(ctx context.Context, store manifest.Store, indexed indexedDocument) error {
	if replacer, ok := store.(chunkReplacer); ok {
		if err := replacer.ReplaceDocumentChunks(ctx, indexed.Document, indexed.Chunks); err != nil {
			return fmt.Errorf("write manifest rows for %s: %w", indexed.Document.Path, err)
		}
		return nil
	}
	if err := store.UpsertDocument(ctx, indexed.Document); err != nil {
		return err
	}
	if err := store.MarkChunksInactive(ctx, indexed.Document.ID); err != nil {
		return err
	}
	for _, row := range indexed.Chunks {
		if err := store.InsertChunk(ctx, row); err != nil {
			return err
		}
	}
	return nil
}

func markCandidateError(ctx context.Context, store manifest.Store, candidate CandidateFile, contentHash string, now time.Time, err error) error {
	errText := err.Error()
	existing, getErr := store.GetDocumentByPath(ctx, candidate.Path)
	if getErr == nil && existing.Status == manifest.DocumentStatusIndexed {
		return store.MarkDocumentError(ctx, candidate.Path, errText)
	}
	if getErr != nil && !errors.Is(getErr, manifest.ErrNotFound) {
		return getErr
	}
	doc := manifest.Document{
		ID:          manifest.DocumentID(candidate.Path),
		Path:        candidate.Path,
		ContentHash: contentHash,
		SizeBytes:   candidate.SizeBytes,
		ModifiedAt:  candidate.ModifiedAt.UTC(),
		Status:      manifest.DocumentStatusError,
		Error:       &errText,
	}
	if doc.ModifiedAt.IsZero() {
		doc.ModifiedAt = now
	}
	return store.UpsertDocument(ctx, doc)
}

func PrintSummary(w io.Writer, summary Summary) {
	fmt.Fprintf(w, "scanned: %d\n", summary.Scanned)
	fmt.Fprintf(w, "new files: %d\n", summary.NewFiles)
	fmt.Fprintf(w, "changed files: %d\n", summary.ChangedFiles)
	fmt.Fprintf(w, "unchanged files: %d\n", summary.UnchangedFiles)
	fmt.Fprintf(w, "deleted files: %d\n", summary.DeletedFiles)
	fmt.Fprintf(w, "failed files: %d\n", summary.FailedFiles)
	fmt.Fprintf(w, "chunks embedded: %d\n", summary.ChunksEmbedded)
	fmt.Fprintf(w, "state dir: %s\n", summary.StateDir)
}

func ParseExtensions(value string) []string {
	parts := strings.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func DefaultExtensionsString() string {
	return strings.Join(DefaultExtensions, ",")
}

func verbosef(opts Options, format string, args ...any) {
	if opts.Verbose {
		fmt.Fprintf(opts.Out, format, args...)
	}
}

func isContextStop(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if ctx != nil && ctx.Err() != nil {
		return true
	}
	return errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

func ingestStoppedError(ctx context.Context, err error) error {
	if ctx != nil && ctx.Err() != nil {
		err = ctx.Err()
	} else if errors.Is(err, context.DeadlineExceeded) {
		err = context.DeadlineExceeded
	} else if errors.Is(err, context.Canceled) {
		err = context.Canceled
	}
	if err == nil {
		err = context.Canceled
	}
	return fmt.Errorf("ingest stopped: %w", err)
}

func clearPendingRunAfterStop(store pendingIndexer, usePendingStore bool, runID string) {
	if !usePendingStore || store == nil || runID == "" {
		return
	}
	_ = store.ClearPendingRun(context.Background(), runID)
}

func progressf(opts Options, format string, args ...any) {
	fmt.Fprintf(opts.Out, format, args...)
}

func progressFileDecision(opts Options, sourceRoot string, path string, decision FileDecision, current int, total int) {
	action := "indexing"
	if opts.DryRun {
		action = "would index"
	}
	if decision == FileDecisionChanged {
		action = "re-indexing"
		if opts.DryRun {
			action = "would re-index"
		}
	}
	display := displayPath(sourceRoot, path)
	emitProgress(opts, ProgressEvent{
		Directory:   sourceRoot,
		CurrentPath: display,
		CurrentFile: current,
		TotalFiles:  total,
		Action:      action,
	})
	progressf(opts, "%s file %d/%d: %s\n", action, current, total, display)
}

func progressChunkEmbedded(opts Options, path string, current int, total int) {
	if total <= 0 || !shouldPrintChunkProgress(current, total) {
		return
	}
	emitProgress(opts, ProgressEvent{
		CurrentPath: path,
		CurrentFile: current,
		TotalFiles:  total,
		Action:      "embedded chunks",
	})
	progressf(opts, "embedded chunks for %s: %d/%d\n", path, current, total)
}

func emitProgress(opts Options, event ProgressEvent) {
	if opts.Progress != nil {
		opts.Progress(event)
	}
}

func shouldPrintChunkProgress(current int, total int) bool {
	if current == 1 || current == total {
		return true
	}
	if total <= 5 {
		return true
	}
	return current%10 == 0
}

func displayPath(root string, path string) string {
	rel, err := filepath.Rel(root, path)
	if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return path
	}
	return rel
}

func deletedDocumentsForSync(ctx context.Context, store manifest.Store, syncSource bool, sourceRoot string, discoveredPaths map[string]struct{}, opts Options) ([]string, error) {
	if !syncSource {
		return nil, nil
	}
	docs, err := store.ListIndexedDocumentsUnderRoot(ctx, sourceRoot)
	if err != nil {
		return nil, fmt.Errorf("list indexed documents under source root: %w", err)
	}
	deletedIDs := make([]string, 0)
	for _, doc := range docs {
		if _, ok := discoveredPaths[doc.Path]; ok {
			continue
		}
		deletedIDs = append(deletedIDs, doc.ID)
		if opts.DryRun {
			verbosef(opts, "would delete missing file: %s\n", doc.Path)
		} else {
			verbosef(opts, "deleting missing file: %s\n", doc.Path)
		}
	}
	return deletedIDs, nil
}

func resolveRoot(root string) (string, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		root = "."
	}
	if !filepath.IsAbs(root) {
		wd, err := os.Getwd()
		if err != nil {
			return "", err
		}
		root = filepath.Join(wd, root)
	}
	return filepath.Clean(root), nil
}

func newRunID() (string, error) {
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(bytes[:]), nil
}
