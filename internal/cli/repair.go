package cli

import (
	"fmt"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/spf13/cobra"
)

func newRepairCommand(opts *options) *cobra.Command {
	var discardPending bool
	var rebuildLexical bool
	cmd := &cobra.Command{
		Use:   "repair",
		Short: "Recover pending ingest runs",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx, cancel := commandContext(cmd, opts)
			defer cancel()

			paths, err := resolvePaths(opts)
			if err != nil {
				return fmt.Errorf("resolve state paths: %w", err)
			}
			lock, err := config.AcquireWriteLock(paths)
			if err != nil {
				return fmt.Errorf("acquire write lock: %w", err)
			}
			defer lock.Release()

			store, err := manifest.Open(paths.ManifestPath)
			if err != nil {
				return fmt.Errorf("open manifest database: %w", err)
			}
			defer store.Close()
			if err := store.Init(ctx); err != nil {
				return fmt.Errorf("initialize manifest database: %w", err)
			}

			runs, err := store.ListPendingRuns(ctx)
			if err != nil {
				return fmt.Errorf("list pending ingest runs: %w", err)
			}
			if len(runs) == 0 && !rebuildLexical {
				fmt.Fprintln(cmd.OutOrStdout(), "No pending ingest runs found.")
				return nil
			}

			for _, run := range runs {
				if discardPending {
					if err := store.ClearPendingRun(ctx, run.ID); err != nil {
						return fmt.Errorf("discard pending ingest run %s: %w", run.ID, err)
					}
					fmt.Fprintf(cmd.OutOrStdout(), "discarded pending ingest run %s (%d documents, %d chunks)\n", run.ID, run.DocumentCount, run.ChunkCount)
					continue
				}
				if err := store.ActivatePendingRun(ctx, run.ID); err != nil {
					return fmt.Errorf("activate pending ingest run %s: %w", run.ID, err)
				}
				fmt.Fprintf(cmd.OutOrStdout(), "activated pending ingest run %s (%d documents, %d chunks)\n", run.ID, run.DocumentCount, run.ChunkCount)
			}
			if rebuildLexical {
				if err := store.RebuildLexicalIndex(ctx); err != nil {
					return fmt.Errorf("rebuild lexical index: %w", err)
				}
				fmt.Fprintln(cmd.OutOrStdout(), "rebuilt lexical index")
			}
			return nil
		},
	}
	cmd.Flags().BoolVar(&discardPending, "discard-pending", false, "discard pending ingest runs instead of activating them")
	cmd.Flags().BoolVar(&rebuildLexical, "rebuild-lexical", false, "rebuild the SQLite lexical index from active indexed chunks")
	return cmd
}
