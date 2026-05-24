package cli

import (
	"context"
	"fmt"
	"os"

	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/spf13/cobra"
)

var newCLIEmbedder = func() (llm.Embedder, error) {
	return llm.NewEnvEmbedder(os.Getenv)
}

var retrieveWithPlan = rag.RetrieveWithPlan

func newQueryCommand(opts *options) *cobra.Command {
	var topK int
	var jsonOutput bool
	var showRetrievalDebug bool
	var openSource int
	var retrievalFlags retrievalFlagValues
	var sourceFlags sourceFilterFlagValues
	cmd := &cobra.Command{
		Use:   "query <question>",
		Short: "Retrieve relevant indexed chunks without calling an LLM",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx, cancel := commandContext(cmd, opts)
			defer cancel()

			paths, err := resolvePaths(opts)
			if err != nil {
				return fmt.Errorf("resolve state paths: %w", err)
			}
			retrievalOpts, err := retrievalOptionsFromFlags(cmd, retrievalFlags)
			if err != nil {
				return err
			}
			sourceFilter, err := sourceFilterFromFlags(sourceFlags)
			if err != nil {
				return err
			}
			retrievalOpts.SourceFilter = sourceFilter
			if err := rag.CheckIndexForMode(paths, retrievalOpts.Mode); err != nil {
				return err
			}
			store, err := manifest.Open(paths.ManifestPath)
			if err != nil {
				return fmt.Errorf("open manifest database: %w", err)
			}
			defer store.Close()
			if err := prepareLexicalIndex(ctx, store, retrievalOpts.Mode); err != nil {
				return err
			}
			if err := configureQueryPlanner(&retrievalOpts); err != nil {
				return err
			}

			var embedder llm.Embedder
			if rag.RetrievalModeUsesVector(retrievalOpts.Mode) {
				embedder, err = newCLIEmbedder()
				if err != nil {
					return fmt.Errorf("configure embedder: %w", err)
				}
			}
			retrievalOpts.Question = args[0]
			retrievalOpts.TopK = topK
			retrievalOpts.Paths = paths
			retrievalOpts.Embedder = embedder
			retrievalOpts, err = rag.NormalizeRetrieveOptions(retrievalOpts)
			if err != nil {
				return err
			}
			result, err := retrieveWithPlan(ctx, store, retrievalOpts)
			if err != nil {
				return err
			}
			if openSource > 0 {
				if err := openRetrievedSource(cmd.ErrOrStderr(), result.Chunks, openSource); err != nil {
					return err
				}
			}
			if jsonOutput {
				return writeQueryJSON(cmd.OutOrStdout(), args[0], retrievalOpts, result)
			}
			if showRetrievalDebug {
				printRetrievalDebug(cmd.OutOrStdout(), retrievalOpts, result.Plan)
			}
			if retrievalFlags.showQueryPlan && !showRetrievalDebug {
				printQueryPlan(cmd.OutOrStdout(), result.Plan)
			}
			printQueryResults(cmd.OutOrStdout(), args[0], result.Chunks)
			return nil
		},
	}
	cmd.Flags().IntVar(&topK, "top-k", 8, "number of chunks to retrieve")
	cmd.Flags().BoolVar(&jsonOutput, "json", false, "print query results as JSON")
	cmd.Flags().BoolVar(&showRetrievalDebug, "show-retrieval-debug", false, "print retrieval configuration and scoring details")
	cmd.Flags().IntVar(&openSource, "open-source", 0, "open retrieved source number in configured editor")
	addRetrievalFlags(cmd, &retrievalFlags)
	addSourceFilterFlags(cmd, &sourceFlags)
	return cmd
}

func prepareLexicalIndex(ctx context.Context, store manifest.Store, mode rag.RetrievalMode) error {
	if !rag.RetrievalModeUsesLexical(mode) {
		return nil
	}
	if err := store.Init(ctx); err != nil {
		return fmt.Errorf("initialize manifest lexical index: %w", err)
	}
	lexicalStore, ok := store.(manifest.LexicalSearchStore)
	if !ok {
		return fmt.Errorf("manifest store does not support lexical search")
	}
	if _, err := lexicalStore.EnsureLexicalIndex(ctx); err != nil {
		return fmt.Errorf("ensure lexical index: %w", err)
	}
	return nil
}
