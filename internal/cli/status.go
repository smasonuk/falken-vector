package cli

import (
	"errors"
	"fmt"
	"os"

	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/spf13/cobra"
)

func newStatusCommand(opts *options) *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Show index health and counts",
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx, cancel := commandContext(cmd, opts)
			defer cancel()

			paths, err := resolvePaths(opts)
			if err != nil {
				return fmt.Errorf("resolve state paths: %w", err)
			}
			if _, err := os.Stat(paths.ManifestPath); err != nil {
				if errors.Is(err, os.ErrNotExist) {
					fmt.Fprintln(cmd.OutOrStdout(), "No index found. Run `falkengo ingest <directory>` first.")
					return nil
				}
				return err
			}
			store, err := manifest.Open(paths.ManifestPath)
			if err != nil {
				return fmt.Errorf("open manifest database: %w", err)
			}
			defer store.Close()
			stats, err := store.Stats(ctx)
			if err != nil {
				return fmt.Errorf("read manifest stats: %w", err)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "State dir: %s\n", paths.StateDir)
			fmt.Fprintf(cmd.OutOrStdout(), "Manifest: %s\n", paths.ManifestPath)
			fmt.Fprintf(cmd.OutOrStdout(), "Vector DB: %s\n\n", paths.VecgoPath)
			fmt.Fprintln(cmd.OutOrStdout(), "Documents:")
			fmt.Fprintf(cmd.OutOrStdout(), "  indexed: %d\n", stats.IndexedDocuments)
			fmt.Fprintf(cmd.OutOrStdout(), "  error: %d\n", stats.ErrorDocuments)
			fmt.Fprintf(cmd.OutOrStdout(), "  deleted: %d\n\n", stats.DeletedDocuments)
			fmt.Fprintln(cmd.OutOrStdout(), "Chunks:")
			fmt.Fprintf(cmd.OutOrStdout(), "  active: %d\n", stats.ActiveChunks)
			fmt.Fprintf(cmd.OutOrStdout(), "  inactive: %d\n\n", stats.InactiveChunks)
			fmt.Fprintln(cmd.OutOrStdout(), "Last indexed:")
			if stats.LastIndexedAt == nil {
				fmt.Fprintln(cmd.OutOrStdout(), "  never")
			} else {
				fmt.Fprintf(cmd.OutOrStdout(), "  %s\n", stats.LastIndexedAt.UTC().Format("2006-01-02T15:04:05Z"))
			}
			return nil
		},
	}
}
