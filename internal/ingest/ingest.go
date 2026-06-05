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
	"sync"
	"time"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/vectorstore"
)

var ErrPendingRunDetected = errors.New("Pending ingest run detected. Run `falkengo repair`.")

type Options struct {
	Root                 string
	Paths                config.Paths
	Extensions           []string
	ExcludeExtensions    []string
	ExcludeDirs          []string
	ChunkSize            int
	ChunkOverlap         int
	ChunkerMode          ChunkerMode
	DryRun               bool
	SyncSource           bool
	Verbose              bool
	Out                  io.Writer
	Embedder             llm.Embedder
	OpenVector           func(context.Context, string, int) (vectorstore.Store, error)
	Now                  func() time.Time
	Progress             func(ProgressEvent)
	EmbeddingConcurrency int
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

type ingestOperation struct {
	ctx                  context.Context
	store                manifest.Store
	opts                 Options
	summary              Summary
	embeddingConcurrency int
	sourceRoot           string
	files                []CandidateFile
	discoveredPaths      map[string]struct{}
	vectorDB             *vectorWriter
	pending              []indexedDocument
	pendingStore         pendingIndexer
	usePendingStore      bool
	runID                string
	jobs                 []indexJob
}

func validateIngestOptions(ctx context.Context, store manifest.Store, opts Options) (Options, int, string, error) {
	embeddingConcurrency, err := normalizeEmbeddingConcurrency(opts.EmbeddingConcurrency)
	if err != nil {
		return opts, 0, "", err
	}
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
	if !opts.DryRun && embeddingConcurrency > 1 {
		opts = synchronizedOptions(opts)
	}
	if !opts.DryRun && opts.Embedder == nil {
		return opts, 0, "", fmt.Errorf("embedder is required")
	}
	sourceRoot, err := resolveRoot(opts.Root)
	if err != nil {
		return opts, 0, "", fmt.Errorf("resolve ingest root: %w", err)
	}
	if !opts.DryRun {
		if pendingStore, ok := store.(manifest.PendingRunStore); ok {
			runs, err := pendingStore.ListPendingRuns(ctx)
			if err != nil {
				if isContextStop(ctx, err) {
					return opts, 0, "", ingestStoppedError(ctx, err)
				}
				return opts, 0, "", fmt.Errorf("list pending ingest runs: %w", err)
			}
			if len(runs) != 0 {
				return opts, 0, "", ErrPendingRunDetected
			}
		}
	}
	return opts, embeddingConcurrency, sourceRoot, nil
}

func (op *ingestOperation) stopErr(err error) error {
	clearPendingRunAfterStop(op.pendingStore, op.usePendingStore, op.runID)
	return ingestStoppedError(op.ctx, err)
}

func (op *ingestOperation) scanFiles() error {
	progressf(op.opts, "scanning %s\n", op.sourceRoot)
	files, err := FindCandidateFiles(op.ctx, WalkOptions{
		Root:              op.sourceRoot,
		StateDir:          op.opts.Paths.StateDir,
		Extensions:        op.opts.Extensions,
		ExcludeExtensions: op.opts.ExcludeExtensions,
		ExcludeDirs:       op.opts.ExcludeDirs,
	})
	if err != nil {
		if isContextStop(op.ctx, err) {
			return op.stopErr(err)
		}
		return fmt.Errorf("walk candidate files: %w", err)
	}
	op.files = files
	op.summary = Summary{Scanned: len(files), StateDir: op.opts.Paths.StateDir}
	progressf(op.opts, "found %d candidate files\n", len(files))
	emitProgress(op.opts, ProgressEvent{
		Directory:  op.sourceRoot,
		Action:     "scanned",
		Scanned:    len(files),
		TotalFiles: len(files),
	})
	return nil
}

func (op *ingestOperation) initRunState() error {
	op.discoveredPaths = make(map[string]struct{}, len(op.files))
	op.vectorDB = &vectorWriter{opts: op.opts}
	op.pending = make([]indexedDocument, 0)
	op.pendingStore, op.usePendingStore = op.store.(pendingIndexer)
	op.runID = ""
	if !op.opts.DryRun && op.usePendingStore {
		runID, err := newRunID()
		if err != nil {
			return fmt.Errorf("create index run id: %w", err)
		}
		op.runID = runID
	}
	op.jobs = make([]indexJob, 0)
	return nil
}

func (op *ingestOperation) processCandidates() error {
	for i, candidate := range op.files {
		if err := op.ctx.Err(); err != nil {
			return op.stopErr(err)
		}
		op.discoveredPaths[candidate.Path] = struct{}{}
		effectiveChunker := EffectiveChunkerMode(candidate.Path, op.opts.ChunkerMode)
		decision, contentHash, err := DecideFileWithConfig(op.ctx, op.store, candidate, ChunkConfig{
			Chunker:          string(effectiveChunker),
			ChunkSize:        op.opts.ChunkSize,
			ChunkOverlap:     op.opts.ChunkOverlap,
			IndexTextVersion: CurrentIndexTextVersion,
		})
		if err != nil {
			if isContextStop(op.ctx, err) {
				return op.stopErr(err)
			}
			op.summary.FailedFiles++
			if !op.opts.DryRun {
				_ = op.store.MarkDocumentError(op.ctx, candidate.Path, err.Error())
			}
			verbosef(op.opts, "failed to inspect %s: %v\n", candidate.Path, err)
			continue
		}
		switch decision {
		case FileDecisionUnchanged:
			op.summary.UnchangedFiles++
			if op.opts.DryRun {
				verbosef(op.opts, "would skip unchanged file: %s\n", candidate.Path)
			} else {
				verbosef(op.opts, "skipping unchanged %s\n", candidate.Path)
			}
			continue
		case FileDecisionNew:
			op.summary.NewFiles++
			if op.opts.DryRun {
				verbosef(op.opts, "would index new file: %s\n", candidate.Path)
			}
		case FileDecisionChanged:
			op.summary.ChangedFiles++
			if op.opts.DryRun {
				verbosef(op.opts, "would re-index changed file: %s\n", candidate.Path)
			}
		}

		progressFileDecision(op.opts, op.sourceRoot, candidate.Path, decision, i+1, len(op.files))
		if !op.opts.DryRun {
			op.jobs = append(op.jobs, indexJob{
				ResultIndex:      len(op.jobs),
				Candidate:        candidate,
				SourceRoot:       op.sourceRoot,
				ContentHash:      contentHash,
				EffectiveChunker: effectiveChunker,
			})
			continue
		}
		indexed, err := indexFile(op.ctx, nil, candidate, op.sourceRoot, contentHash, effectiveChunker, op.opts)
		if err != nil {
			if isContextStop(op.ctx, err) {
				return op.stopErr(err)
			}
			op.summary.FailedFiles++
			verbosef(op.opts, "failed to index %s: %v\n", candidate.Path, err)
			continue
		}
		op.pending = append(op.pending, indexed)
		op.summary.ChunksEmbedded += len(indexed.Chunks)
	}
	return nil
}

func (op *ingestOperation) processJobs() error {
	if op.opts.DryRun || len(op.jobs) == 0 {
		return nil
	}
	results, err := indexJobs(op.ctx, op.jobs, op.vectorDB, op.opts, op.embeddingConcurrency)
	if err != nil {
		return op.stopErr(err)
	}
	for _, result := range results {
		candidate := result.Job.Candidate
		if result.Err != nil {
			if isContextStop(op.ctx, result.Err) {
				return op.stopErr(result.Err)
			}
			op.summary.FailedFiles++
			_ = markCandidateError(op.ctx, op.store, candidate, result.Job.ContentHash, op.opts.Now(), result.Err)
			verbosef(op.opts, "failed to index %s: %v\n", candidate.Path, result.Err)
			continue
		}
		indexed := result.Indexed
		if op.usePendingStore {
			if err := op.pendingStore.StagePendingDocument(op.ctx, op.runID, indexed.Document, indexed.Chunks); err != nil {
				if isContextStop(op.ctx, err) {
					return op.stopErr(err)
				}
				_ = op.pendingStore.ClearPendingRun(context.Background(), op.runID)
				return fmt.Errorf("stage manifest rows for %s: %w", indexed.Document.Path, err)
			}
		} else {
			op.pending = append(op.pending, indexed)
		}
		op.summary.ChunksEmbedded += len(indexed.Chunks)
	}
	return nil
}

func (op *ingestOperation) finalizeRun() error {
	if err := op.ctx.Err(); err != nil {
		return op.stopErr(err)
	}
	deletedIDs, err := deletedDocumentsForSync(op.ctx, op.store, op.opts.SyncSource, op.sourceRoot, op.discoveredPaths, op.opts)
	if err != nil {
		if isContextStop(op.ctx, err) {
			return op.stopErr(err)
		}
		return err
	}
	op.summary.DeletedFiles = len(deletedIDs)

	if op.vectorDB.HasStore() && !op.opts.DryRun {
		if err := op.ctx.Err(); err != nil {
			return op.stopErr(err)
		}
		progressf(op.opts, "committing vector database\n")
		if err := op.vectorDB.Commit(op.ctx); err != nil {
			if isContextStop(op.ctx, err) {
				return op.stopErr(err)
			}
			if op.usePendingStore {
				_ = op.pendingStore.ClearPendingRun(context.Background(), op.runID)
			}
			return fmt.Errorf("commit vector database: %w", err)
		}
	}
	if !op.opts.DryRun {
		if err := op.ctx.Err(); err != nil {
			return op.stopErr(err)
		}
		progressf(op.opts, "updating manifest\n")
		if op.usePendingStore {
			if err := op.pendingStore.ActivatePendingRun(op.ctx, op.runID); err != nil {
				if isContextStop(op.ctx, err) {
					return op.stopErr(err)
				}
				return fmt.Errorf("activate staged manifest rows: %w", err)
			}
		} else {
			for _, indexed := range op.pending {
				if err := writeIndexedDocument(op.ctx, op.store, indexed); err != nil {
					if isContextStop(op.ctx, err) {
						return op.stopErr(err)
					}
					return err
				}
			}
		}
		if len(deletedIDs) != 0 {
			if err := op.store.MarkDocumentsDeleted(op.ctx, deletedIDs, op.opts.Now()); err != nil {
				if isContextStop(op.ctx, err) {
					return op.stopErr(err)
				}
				return fmt.Errorf("mark deleted source documents: %w", err)
			}
		}
	}
	processedCandidates := op.summary.Scanned - op.summary.UnchangedFiles
	if processedCandidates > 0 && op.summary.FailedFiles == processedCandidates {
		return fmt.Errorf("all files failed to index")
	}
	return nil
}

func Run(ctx context.Context, store manifest.Store, opts Options) (Summary, error) {
	opts, embeddingConcurrency, sourceRoot, err := validateIngestOptions(ctx, store, opts)
	if err != nil {
		return Summary{}, err
	}

	op := &ingestOperation{
		ctx:                  ctx,
		store:                store,
		opts:                 opts,
		embeddingConcurrency: embeddingConcurrency,
		sourceRoot:           sourceRoot,
	}

	if err := op.scanFiles(); err != nil {
		return op.summary, err
	}
	if err := op.initRunState(); err != nil {
		return op.summary, err
	}
	defer func() {
		if op.vectorDB != nil {
			_ = op.vectorDB.Close()
		}
	}()

	if err := op.processCandidates(); err != nil {
		return op.summary, err
	}
	if err := op.processJobs(); err != nil {
		return op.summary, err
	}
	if err := op.finalizeRun(); err != nil {
		return op.summary, err
	}

	return op.summary, nil
}

type indexedDocument struct {
	Document manifest.Document
	Chunks   []manifest.Chunk
}

type indexJob struct {
	ResultIndex      int
	Candidate        CandidateFile
	SourceRoot       string
	ContentHash      string
	EffectiveChunker ChunkerMode
}

type indexResult struct {
	Job     indexJob
	Indexed indexedDocument
	Err     error
}

func indexJobs(ctx context.Context, jobs []indexJob, vectorDB *vectorWriter, opts Options, concurrency int) ([]indexResult, error) {
	if len(jobs) == 0 {
		return nil, nil
	}
	if concurrency <= 1 {
		results := make([]indexResult, 0, len(jobs))
		for _, job := range jobs {
			if err := ctx.Err(); err != nil {
				return results, err
			}
			indexed, err := indexFile(ctx, vectorDB, job.Candidate, job.SourceRoot, job.ContentHash, job.EffectiveChunker, opts)
			results = append(results, indexResult{Job: job, Indexed: indexed, Err: err})
		}
		return results, nil
	}
	if concurrency > len(jobs) {
		concurrency = len(jobs)
	}

	jobCh := make(chan indexJob)
	resultCh := make(chan indexResult, len(jobs))
	var wg sync.WaitGroup
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for job := range jobCh {
				if err := ctx.Err(); err != nil {
					resultCh <- indexResult{Job: job, Err: err}
					continue
				}
				indexed, err := indexFile(ctx, vectorDB, job.Candidate, job.SourceRoot, job.ContentHash, job.EffectiveChunker, opts)
				resultCh <- indexResult{Job: job, Indexed: indexed, Err: err}
			}
		}()
	}

	sendErr := error(nil)
	for _, job := range jobs {
		if sendErr != nil {
			break
		}
		select {
		case <-ctx.Done():
			sendErr = ctx.Err()
		case jobCh <- job:
		}
	}
	close(jobCh)
	wg.Wait()
	close(resultCh)

	results := make([]indexResult, len(jobs))
	received := 0
	for result := range resultCh {
		results[result.Job.ResultIndex] = result
		received++
	}
	if sendErr != nil {
		return results[:received], sendErr
	}
	return results, nil
}

type vectorWriter struct {
	mu    sync.Mutex
	store vectorstore.Store
	opts  Options
}

func (w *vectorWriter) Insert(ctx context.Context, vector []float32, chunkID string) (int64, error) {
	if w == nil {
		return 0, errors.New("vector writer is required")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.store == nil {
		store, err := w.opts.OpenVector(ctx, w.opts.Paths.VecgoPath, len(vector))
		if err != nil {
			return 0, fmt.Errorf("open vector database: %w", err)
		}
		w.store = store
	}
	return w.store.Insert(ctx, vector, chunkID)
}

func (w *vectorWriter) HasStore() bool {
	if w == nil {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.store != nil
}

func (w *vectorWriter) Commit(ctx context.Context) error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.store == nil {
		return nil
	}
	return w.store.Commit(ctx)
}

func (w *vectorWriter) Close() error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.store == nil {
		return nil
	}
	err := w.store.Close()
	w.store = nil
	return err
}

func indexFile(ctx context.Context, vectorDB *vectorWriter, candidate CandidateFile, sourceRoot string, contentHash string, effectiveChunker ChunkerMode, opts Options) (indexedDocument, error) {
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
		chunkID := manifest.ChunkID(docID, chunk.Index, chunk.Hash)
		vectorID, err := vectorDB.Insert(ctx, embedding.Vector, chunkID)
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
	return ParseCSVList(value)
}

func ParseCSVList(value string) []string {
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

func normalizeEmbeddingConcurrency(value int) (int, error) {
	if value < 0 {
		return 0, fmt.Errorf("embedding concurrency must be >= 0")
	}
	if value == 0 {
		return config.DefaultEmbeddingConcurrency, nil
	}
	return value, nil
}

type lockedWriter struct {
	mu   *sync.Mutex
	next io.Writer
}

func (w lockedWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.next.Write(p)
}

func synchronizedOptions(opts Options) Options {
	var outMu sync.Mutex
	opts.Out = lockedWriter{mu: &outMu, next: opts.Out}
	if opts.Progress != nil {
		progress := opts.Progress
		var progressMu sync.Mutex
		opts.Progress = func(event ProgressEvent) {
			progressMu.Lock()
			defer progressMu.Unlock()
			progress(event)
		}
	}
	return opts
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
