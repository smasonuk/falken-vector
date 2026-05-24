package cli

import (
	"errors"
	"fmt"
	"os"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/ingest"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/spf13/cobra"
)

func newIngestCommand(opts *options) *cobra.Command {
	var extensions string
	var chunkSize int
	var chunkOverlap int
	var chunker string
	var dryRun bool
	var syncSource bool
	cmd := &cobra.Command{
		Use:   "ingest <directory>",
		Short: "Index text files from a directory",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := longRunningCommandContext(cmd, opts)
			defer cancel()

			paths, err := resolvePaths(opts)
			if err != nil {
				return fmt.Errorf("resolve state paths: %w", err)
			}
			var store manifest.Store
			if dryRun {
				if _, err := os.Stat(paths.ManifestPath); err == nil {
					sqliteStore, err := manifest.Open(paths.ManifestPath)
					if err != nil {
						return fmt.Errorf("open manifest database: %w", err)
					}
					defer sqliteStore.Close()
					store = sqliteStore
				} else if errors.Is(err, os.ErrNotExist) {
					store = manifest.EmptyStore{}
				} else {
					return fmt.Errorf("inspect manifest database: %w", err)
				}
			} else {
				lock, err := config.AcquireWriteLock(paths)
				if err != nil {
					return fmt.Errorf("acquire write lock: %w", err)
				}
				defer lock.Release()
				if err := config.EnsureStateDirs(paths); err != nil {
					return fmt.Errorf("create state directories: %w", err)
				}

				sqliteStore, err := manifest.Open(paths.ManifestPath)
				if err != nil {
					return fmt.Errorf("open manifest database: %w", err)
				}
				defer sqliteStore.Close()
				if err := sqliteStore.Init(ctx); err != nil {
					return fmt.Errorf("initialize manifest database: %w", err)
				}
				store = sqliteStore
			}

			var embedder llm.Embedder
			if !dryRun {
				embedder = &lazyEnvEmbedder{}
			}
			chunkerMode, err := ingest.ParseChunkerMode(chunker)
			if err != nil {
				return err
			}
			summary, err := ingest.Run(ctx, store, ingest.Options{
				Root:         args[0],
				Paths:        paths,
				Extensions:   ingest.ParseExtensions(extensions),
				ChunkSize:    chunkSize,
				ChunkOverlap: chunkOverlap,
				ChunkerMode:  chunkerMode,
				DryRun:       dryRun,
				SyncSource:   syncSource,
				Verbose:      opts.verbose,
				Out:          cmd.OutOrStdout(),
				Embedder:     embedder,
			})
			ingest.PrintSummary(cmd.OutOrStdout(), summary)
			return err
		},
	}
	cmd.Flags().StringVar(&extensions, "extensions", "", "restrict indexing to comma-separated file extensions")
	cmd.Flags().StringVar(&chunker, "chunker", "auto", "chunking strategy: auto, fixed, markdown, text, or code")
	cmd.Flags().IntVar(&chunkSize, "chunk-size", 1200, "target maximum chunk size in characters")
	cmd.Flags().IntVar(&chunkOverlap, "chunk-overlap", 200, "overlap in characters for fixed chunking and large-section fallback")
	cmd.Flags().BoolVar(&dryRun, "dry-run", false, "scan and classify files without writing vectors or manifest chunks")
	cmd.Flags().BoolVar(&syncSource, "sync-source", false, "mark indexed files missing from this source directory as deleted")
	return cmd
}
