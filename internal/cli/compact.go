package cli

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/smasonuk/falken-vector/internal/compact"
	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/spf13/cobra"
)

func newCompactCommand(opts *options) *cobra.Command {
	var dryRun bool
	var keepBackup bool
	var batchSize int
	var embeddingConcurrency int
	cmd := &cobra.Command{
		Use:   "compact",
		Short: "Rebuild the vector database from active manifest chunks",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx, cancel := longRunningCommandContext(cmd, opts)
			defer cancel()
			if embeddingConcurrency < 1 {
				return fmt.Errorf("--embedding-concurrency must be >= 1")
			}

			paths, err := resolvePaths(opts)
			if err != nil {
				return fmt.Errorf("resolve state paths: %w", err)
			}
			if _, err := os.Stat(paths.ManifestPath); err != nil {
				if errors.Is(err, os.ErrNotExist) {
					return rag.ErrNoIndex
				}
				return fmt.Errorf("inspect manifest database: %w", err)
			}

			var lock *config.WriteLock
			if !dryRun {
				lock, err = config.AcquireWriteLock(paths)
				if err != nil {
					return fmt.Errorf("acquire write lock: %w", err)
				}
				defer lock.Release()
			}

			store, err := manifest.Open(paths.ManifestPath)
			if err != nil {
				return fmt.Errorf("open manifest database: %w", err)
			}
			defer store.Close()
			if !dryRun {
				if err := store.Init(ctx); err != nil {
					return fmt.Errorf("initialize manifest database: %w", err)
				}
			}

			summary, err := compact.Run(ctx, store, compact.Options{
				Paths:                paths,
				Embedder:             &lazyEnvEmbedder{},
				DryRun:               dryRun,
				KeepBackup:           keepBackup,
				BatchSize:            batchSize,
				Verbose:              opts.verbose,
				Out:                  cmd.OutOrStdout(),
				EmbeddingConcurrency: embeddingConcurrency,
			})
			if err == nil || dryRun || summary.ActiveChunks > 0 || summary.ReembeddedChunks > 0 {
				compact.PrintSummary(cmd.OutOrStdout(), summary, dryRun)
			}
			return err
		},
	}
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "show what would be compacted without writing vectors or manifest updates")
	cmd.Flags().BoolVar(&keepBackup, "keep-backup", false, "keep the old vector database backup after a successful compact")
	cmd.Flags().IntVar(&batchSize, "batch-size", 100, "number of manifest chunks to fetch and process per batch")
	cmd.Flags().IntVar(&embeddingConcurrency, "embedding-concurrency", config.DefaultEmbeddingConcurrency, "number of concurrent embedding requests during compaction")
	return cmd
}

type lazyEnvEmbedder struct {
	once     sync.Once
	embedder llm.Embedder
	err      error
}

func (e *lazyEnvEmbedder) EmbedText(ctx context.Context, input string) (llm.Embedding, error) {
	e.once.Do(func() {
		e.embedder, e.err = llm.NewEnvEmbedder(os.Getenv)
	})
	if e.err != nil {
		return llm.Embedding{}, e.err
	}
	return e.embedder.EmbedText(ctx, input)
}
