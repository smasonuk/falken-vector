package cli

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/spf13/cobra"
)

func newResetCommand(opts *options) *cobra.Command {
	var yes bool
	var forceStateDir bool
	cmd := &cobra.Command{
		Use:   "reset",
		Short: "Delete local falkengo state",
		RunE: func(cmd *cobra.Command, _ []string) error {
			paths, err := resolvePaths(opts)
			if err != nil {
				return fmt.Errorf("resolve state paths: %w", err)
			}
			if err := config.ValidateResetStateDir(paths, forceStateDir); err != nil {
				return err
			}
			if !yes {
				fmt.Fprintf(cmd.OutOrStdout(), "This will delete %s including the manifest and vector database. Continue? [y/N] ", paths.StateDir)
				reader := bufio.NewReader(cmd.InOrStdin())
				line, err := reader.ReadString('\n')
				if err != nil && strings.TrimSpace(line) == "" {
					return fmt.Errorf("read confirmation: %w", err)
				}
				answer := strings.ToLower(strings.TrimSpace(line))
				if answer != "y" && answer != "yes" {
					fmt.Fprintln(cmd.OutOrStdout(), "reset cancelled")
					return nil
				}
			}
			lock, err := config.AcquireWriteLock(paths)
			if err != nil {
				return fmt.Errorf("acquire write lock: %w", err)
			}
			defer lock.Release()
			if err := os.RemoveAll(paths.StateDir); err != nil {
				return fmt.Errorf("delete state dir: %w", err)
			}
			fmt.Fprintf(cmd.OutOrStdout(), "deleted %s\n", paths.StateDir)
			return nil
		},
	}
	cmd.Flags().BoolVar(&yes, "yes", false, "delete without prompting")
	cmd.Flags().BoolVar(&forceStateDir, "force-state-dir", false, "allow reset when --state-dir basename is not .falkengo")
	return cmd
}
