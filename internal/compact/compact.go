package compact

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

var ErrPendingRunDetected = errors.New("pending ingest run detected; run `falkengo repair` before compacting")

type Options struct {
	Paths                config.Paths
	Embedder             llm.Embedder
	OpenVector           func(context.Context, string, int) (vectorstore.Store, error)
	DryRun               bool
	KeepBackup           bool
	BatchSize            int
	Verbose              bool
	Out                  io.Writer
	Now                  func() time.Time
	EmbeddingConcurrency int
}

type Summary struct {
	ActiveChunks     int
	InactiveChunks   int
	ReembeddedChunks int
	UpdatedChunks    int
	PendingRuns      int
	EmbeddingModel   string
	StateDir         string
	VectorPath       string
	BackupPath       string
}

func Run(ctx context.Context, store manifest.Store, opts Options) (Summary, error) {
	embeddingConcurrency, err := normalizeEmbeddingConcurrency(opts.EmbeddingConcurrency)
	if err != nil {
		return Summary{}, err
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = 100
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
	summary := Summary{
		StateDir:   opts.Paths.StateDir,
		VectorPath: opts.Paths.VecgoPath,
	}

	if pendingStore, ok := store.(manifest.PendingRunStore); ok {
		runs, err := pendingStore.ListPendingRuns(ctx)
		if err != nil {
			return summary, fmt.Errorf("list pending ingest runs: %w", err)
		}
		summary.PendingRuns = len(runs)
		if len(runs) != 0 {
			if !opts.DryRun {
				return summary, ErrPendingRunDetected
			}
		}
	}

	stats, err := store.Stats(ctx)
	if err != nil {
		return summary, fmt.Errorf("read manifest stats: %w", err)
	}
	summary.InactiveChunks = stats.InactiveChunks

	compactStore, ok := store.(manifest.CompactStore)
	if !ok {
		return summary, errors.New("manifest store does not support compaction")
	}
	if opts.DryRun {
		activeChunks, err := forEachActiveIndexedChunkBatch(ctx, compactStore, opts.BatchSize, func([]manifest.Chunk) error {
			return nil
		})
		if err != nil {
			return summary, fmt.Errorf("list active indexed chunks: %w", err)
		}
		summary.ActiveChunks = activeChunks
		return summary, nil
	}
	if err := os.MkdirAll(opts.Paths.StateDir, 0o755); err != nil {
		return summary, fmt.Errorf("create state directory: %w", err)
	}

	runID, err := newRunID()
	if err != nil {
		return summary, fmt.Errorf("create compact run id: %w", err)
	}
	tempPath := filepath.Join(opts.Paths.StateDir, config.VecgoDirName+".compact-"+runID+".tmp")
	if err := os.RemoveAll(tempPath); err != nil {
		return summary, fmt.Errorf("clear compact temp vector database: %w", err)
	}
	defer func() {
		_ = os.RemoveAll(tempPath)
	}()

	updates, activeChunks, err := rebuildTempVectorDB(ctx, tempPath, compactStore, opts, embeddingConcurrency)
	if err != nil {
		return summary, err
	}
	summary.ActiveChunks = activeChunks
	if activeChunks == 0 {
		return summary, nil
	}
	summary.ReembeddedChunks = len(updates)
	if len(updates) != 0 {
		summary.EmbeddingModel = updates[0].EmbeddingModel
	}

	backupPath, err := replaceVectorDB(opts.Paths.VecgoPath, tempPath, runID)
	if err != nil {
		return summary, err
	}
	if backupPath != "" {
		summary.BackupPath = backupPath
	}

	if err := compactStore.UpdateChunkVectorRefs(ctx, updates); err != nil {
		if backupPath != "" {
			return summary, fmt.Errorf("vector database was replaced but manifest vector refs were not fully updated: %w; backup kept at %s", err, backupPath)
		}
		return summary, fmt.Errorf("vector database was replaced but manifest vector refs were not fully updated: %w", err)
	}
	summary.UpdatedChunks = len(updates)
	if backupPath != "" && !opts.KeepBackup {
		if err := os.RemoveAll(backupPath); err != nil {
			summary.BackupPath = backupPath
			verbosef(opts, "failed to remove compact backup %s: %v\n", backupPath, err)
			return summary, nil
		}
		summary.BackupPath = ""
	}
	return summary, nil
}

func PrintSummary(w io.Writer, summary Summary, dryRun bool) {
	if w == nil {
		w = io.Discard
	}
	if summary.ActiveChunks == 0 {
		fmt.Fprintf(w, "active chunks: %d\n", summary.ActiveChunks)
		fmt.Fprintln(w, "nothing to compact")
		printPendingRunDryRunWarning(w, summary, dryRun)
		fmt.Fprintf(w, "state dir: %s\n", summary.StateDir)
		return
	}
	if dryRun {
		fmt.Fprintln(w, "compact dry run")
		fmt.Fprintf(w, "active chunks: %d\n", summary.ActiveChunks)
		fmt.Fprintf(w, "inactive chunks: %d\n", summary.InactiveChunks)
		fmt.Fprintf(w, "would rebuild vector database: %s\n", summary.VectorPath)
		fmt.Fprintf(w, "would re-embed chunks: %d\n", summary.ActiveChunks)
		printPendingRunDryRunWarning(w, summary, dryRun)
		fmt.Fprintf(w, "state dir: %s\n", summary.StateDir)
		return
	}
	fmt.Fprintln(w, "compacted vector database")
	fmt.Fprintf(w, "active chunks re-embedded: %d\n", summary.ReembeddedChunks)
	fmt.Fprintf(w, "embedding model: %s\n", summary.EmbeddingModel)
	fmt.Fprintf(w, "vector db: %s\n", summary.VectorPath)
	if summary.BackupPath != "" {
		fmt.Fprintf(w, "backup: %s\n", summary.BackupPath)
	}
	fmt.Fprintf(w, "state dir: %s\n", summary.StateDir)
}

func printPendingRunDryRunWarning(w io.Writer, summary Summary, dryRun bool) {
	if dryRun && summary.PendingRuns > 0 {
		fmt.Fprintln(w, "warning: pending ingest run detected; real compact is blocked until `falkengo repair` is run")
	}
}

func forEachActiveIndexedChunkBatch(ctx context.Context, store manifest.CompactStore, batchSize int, fn func([]manifest.Chunk) error) (int, error) {
	total := 0
	for offset := 0; ; offset += batchSize {
		batch, err := store.ListActiveIndexedChunks(ctx, batchSize, offset)
		if err != nil {
			return total, err
		}
		if len(batch) == 0 {
			return total, nil
		}
		total += len(batch)
		if err := fn(batch); err != nil {
			return total, err
		}
	}
}

func rebuildTempVectorDB(ctx context.Context, tempPath string, store manifest.CompactStore, opts Options, embeddingConcurrency int) ([]manifest.ChunkVectorUpdate, int, error) {
	var vectorDB vectorstore.Store
	closeVector := func() error {
		if vectorDB == nil {
			return nil
		}
		err := vectorDB.Close()
		vectorDB = nil
		return err
	}
	defer closeVector()

	updates := make([]manifest.ChunkVectorUpdate, 0)
	dimensions := 0
	activeChunks, err := forEachActiveIndexedChunkBatch(ctx, store, opts.BatchSize, func(batch []manifest.Chunk) error {
		embedded, err := embedChunkBatch(ctx, batch, opts, embeddingConcurrency)
		if err != nil {
			return err
		}
		for _, item := range embedded {
			chunk := item.Chunk
			embedding := item.Embedding
			if len(embedding.Vector) == 0 {
				return fmt.Errorf("embed chunk %s: empty vector", chunk.ID)
			}
			if dimensions == 0 {
				dimensions = len(embedding.Vector)
				store, err := opts.OpenVector(ctx, tempPath, dimensions)
				if err != nil {
					return fmt.Errorf("open compact vector database: %w", err)
				}
				vectorDB = store
			} else if len(embedding.Vector) != dimensions {
				return fmt.Errorf("embedding dimension mismatch for chunk %s: got %d, want %d", chunk.ID, len(embedding.Vector), dimensions)
			}
			vectorID, err := vectorDB.Insert(ctx, embedding.Vector, chunk.ID)
			if err != nil {
				return fmt.Errorf("insert compact vector for chunk %s: %w", chunk.ID, err)
			}
			updates = append(updates, manifest.ChunkVectorUpdate{
				ChunkID:        chunk.ID,
				VectorID:       vectorID,
				EmbeddingModel: embedding.Model,
			})
		}
		return nil
	})
	if err != nil {
		return nil, activeChunks, err
	}
	if vectorDB == nil {
		return updates, activeChunks, nil
	}
	if err := vectorDB.Commit(ctx); err != nil {
		return nil, activeChunks, fmt.Errorf("commit compact vector database: %w", err)
	}
	if err := closeVector(); err != nil {
		return nil, activeChunks, fmt.Errorf("close compact vector database: %w", err)
	}
	return updates, activeChunks, nil
}

type embeddedChunk struct {
	Chunk     manifest.Chunk
	Embedding llm.Embedding
}

type embedChunkResult struct {
	Index     int
	Embedding llm.Embedding
	Err       error
}

func embedChunkBatch(ctx context.Context, batch []manifest.Chunk, opts Options, concurrency int) ([]embeddedChunk, error) {
	if opts.Embedder == nil {
		return nil, errors.New("embedder is required")
	}
	if len(batch) == 0 {
		return nil, nil
	}
	if concurrency <= 1 {
		out := make([]embeddedChunk, 0, len(batch))
		for _, chunk := range batch {
			verbosef(opts, "re-embedding chunk %s\n", chunk.ID)
			embedding, err := opts.Embedder.EmbedText(ctx, textForEmbedding(chunk))
			if err != nil {
				return nil, fmt.Errorf("embed chunk %s: %w", chunk.ID, err)
			}
			out = append(out, embeddedChunk{Chunk: chunk, Embedding: embedding})
		}
		return out, nil
	}
	if concurrency > len(batch) {
		concurrency = len(batch)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobCh := make(chan int)
	resultCh := make(chan embedChunkResult, len(batch))
	var wg sync.WaitGroup
	var verboseMu sync.Mutex
	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range jobCh {
				chunk := batch[index]
				verboseMu.Lock()
				verbosef(opts, "re-embedding chunk %s\n", chunk.ID)
				verboseMu.Unlock()
				embedding, err := opts.Embedder.EmbedText(ctx, textForEmbedding(chunk))
				if err != nil {
					resultCh <- embedChunkResult{Index: index, Err: fmt.Errorf("embed chunk %s: %w", chunk.ID, err)}
					cancel()
					continue
				}
				resultCh <- embedChunkResult{Index: index, Embedding: embedding}
			}
		}()
	}

	sendErr := error(nil)
	for index := range batch {
		if sendErr != nil {
			break
		}
		select {
		case <-ctx.Done():
			sendErr = ctx.Err()
		case jobCh <- index:
		}
	}
	close(jobCh)
	wg.Wait()
	close(resultCh)

	out := make([]embeddedChunk, len(batch))
	received := 0
	var firstErr error
	for result := range resultCh {
		received++
		if result.Err != nil && firstErr == nil {
			firstErr = result.Err
			continue
		}
		out[result.Index] = embeddedChunk{Chunk: batch[result.Index], Embedding: result.Embedding}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	if sendErr != nil {
		return nil, sendErr
	}
	return out[:received], nil
}

func textForEmbedding(chunk manifest.Chunk) string {
	if text := strings.TrimSpace(chunk.IndexedText); text != "" {
		return chunk.IndexedText
	}
	return chunk.ChunkText
}

func replaceVectorDB(vectorPath string, tempPath string, runID string) (string, error) {
	backupPath := vectorPath + ".backup-" + runID
	hasExisting, err := pathExists(vectorPath)
	if err != nil {
		return "", fmt.Errorf("inspect existing vector database: %w", err)
	}
	if hasExisting {
		if err := os.Rename(vectorPath, backupPath); err != nil {
			return "", fmt.Errorf("move existing vector database to backup: %w", err)
		}
	}
	if err := os.Rename(tempPath, vectorPath); err != nil {
		if hasExisting {
			if restoreErr := os.Rename(backupPath, vectorPath); restoreErr != nil {
				return backupPath, fmt.Errorf("replace vector database: %w; restore backup failed: %v", err, restoreErr)
			}
			return "", fmt.Errorf("replace vector database: %w; restored backup %s", err, backupPath)
		}
		return "", fmt.Errorf("replace vector database: %w", err)
	}
	if hasExisting {
		return backupPath, nil
	}
	return "", nil
}

func pathExists(path string) (bool, error) {
	if _, err := os.Stat(path); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	return true, nil
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

func verbosef(opts Options, format string, args ...any) {
	if opts.Verbose {
		fmt.Fprintf(opts.Out, format, args...)
	}
}

func newRunID() (string, error) {
	var bytes [16]byte
	if _, err := rand.Read(bytes[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(bytes[:]), nil
}
