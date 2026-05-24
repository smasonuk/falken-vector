package cli

import (
	"context"
	"time"

	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/spf13/cobra"
)

type options struct {
	stateDir string
	timeout  time.Duration
	verbose  bool
}

func Execute() error {
	return NewRootCommand().Execute()
}

func NewRootCommand() *cobra.Command {
	opts := &options{
		stateDir: config.DefaultStateDir,
		timeout:  config.DefaultTimeout,
	}
	cmd := &cobra.Command{
		Use:               "falkengo",
		Short:             "Project-local RAG over text files",
		SilenceUsage:      true,
		SilenceErrors:     true,
		CompletionOptions: cobra.CompletionOptions{DisableDefaultCmd: true},
	}
	cmd.PersistentFlags().StringVar(&opts.stateDir, "state-dir", config.DefaultStateDir, "local application state directory")
	cmd.PersistentFlags().DurationVar(&opts.timeout, "timeout", config.DefaultTimeout, "operation timeout")
	cmd.PersistentFlags().BoolVar(&opts.verbose, "verbose", false, "print detailed progress")

	cmd.AddCommand(newIngestCommand(opts))
	cmd.AddCommand(newQueryCommand(opts))
	cmd.AddCommand(newAskCommand(opts))
	cmd.AddCommand(newEvalCommand(opts))
	cmd.AddCommand(newStatusCommand(opts))
	cmd.AddCommand(newCompactCommand(opts))
	cmd.AddCommand(newResetCommand(opts))
	cmd.AddCommand(newRepairCommand(opts))
	cmd.AddCommand(newVersionCommand())
	return cmd
}

func commandContext(cmd *cobra.Command, opts *options) (context.Context, context.CancelFunc) {
	ctx := cmd.Context()
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithTimeout(ctx, opts.timeout)
}

func resolvePaths(opts *options) (config.Paths, error) {
	return config.ResolvePaths(opts.stateDir)
}
