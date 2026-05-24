package cli

import (
	"fmt"
	"os"

	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/smasonuk/falken-vector/internal/retrievaleval"
	"github.com/spf13/cobra"
)

var (
	newEvalEmbedder = func() (llm.Embedder, error) {
		return llm.NewEnvEmbedder(os.Getenv)
	}
	runRetrievalEval = retrievaleval.Run
)

func newEvalCommand(opts *options) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "eval",
		Short: "Evaluate retrieval quality",
	}
	cmd.AddCommand(newEvalRetrievalCommand(opts))
	return cmd
}

func newEvalRetrievalCommand(opts *options) *cobra.Command {
	var datasetPath string
	var topK int
	var format string
	var details bool
	var failUnderHit float64
	var failUnderRecall float64
	var retrievalFlags retrievalFlagValues
	cmd := &cobra.Command{
		Use:   "retrieval",
		Short: "Evaluate retrieval quality against a JSONL dataset",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			ctx, cancel := commandContext(cmd, opts)
			defer cancel()

			if datasetPath == "" {
				return fmt.Errorf("--dataset is required")
			}
			if topK <= 0 {
				return fmt.Errorf("--top-k must be > 0")
			}
			if format != "text" && format != "json" {
				return fmt.Errorf("--format must be text or json")
			}
			retrievalOpts, err := retrievalOptionsFromFlags(cmd, retrievalFlags)
			if err != nil {
				return err
			}
			includeDetails := details || retrievalFlags.showQueryPlan
			if err := validateThreshold("--fail-under-hit", failUnderHit); err != nil {
				return err
			}
			if err := validateThreshold("--fail-under-recall", failUnderRecall); err != nil {
				return err
			}

			paths, err := resolvePaths(opts)
			if err != nil {
				return fmt.Errorf("resolve state paths: %w", err)
			}
			if err := rag.CheckIndexForMode(paths, retrievalOpts.Mode); err != nil {
				return err
			}
			cases, err := loadRetrievalEvalDataset(datasetPath)
			if err != nil {
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
				embedder, err = newEvalEmbedder()
				if err != nil {
					return fmt.Errorf("configure embedder: %w", err)
				}
			}
			summary, err := runRetrievalEval(ctx, store, cases, retrievaleval.Options{
				Paths:             paths,
				TopK:              topK,
				CandidateK:        retrievalOpts.CandidateK,
				VectorCandidateK:  retrievalOpts.VectorCandidateK,
				LexicalCandidateK: retrievalOpts.LexicalCandidateK,
				Mode:              retrievalOpts.Mode,
				RerankerMode:      retrievalOpts.RerankerMode,
				QueryPlannerMode:  retrievalOpts.QueryPlannerMode,
				QueryPlanner:      retrievalOpts.QueryPlanner,
				MaxSubqueries:     retrievalOpts.MaxSubqueries,
				Embedder:          embedder,
				Details:           includeDetails,
			})
			if err != nil {
				return err
			}
			if format == "json" {
				if err := retrievaleval.WriteJSON(cmd.OutOrStdout(), summary); err != nil {
					return fmt.Errorf("write retrieval evaluation JSON: %w", err)
				}
			} else {
				if err := retrievaleval.WriteText(cmd.OutOrStdout(), datasetPath, summary, includeDetails); err != nil {
					return fmt.Errorf("write retrieval evaluation report: %w", err)
				}
			}
			if failUnderHit > 0 && summary.HitAtK < failUnderHit {
				return fmt.Errorf("retrieval evaluation failed: hit@%d %.4f below threshold %.4f", summary.TopK, summary.HitAtK, failUnderHit)
			}
			if failUnderRecall > 0 && summary.RecallAtK < failUnderRecall {
				return fmt.Errorf("retrieval evaluation failed: recall@%d %.4f below threshold %.4f", summary.TopK, summary.RecallAtK, failUnderRecall)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&datasetPath, "dataset", "", "JSONL retrieval evaluation dataset")
	cmd.Flags().IntVar(&topK, "top-k", 8, "number of chunks to retrieve per case")
	cmd.Flags().StringVar(&format, "format", "text", "output format: text or json")
	cmd.Flags().BoolVar(&details, "details", false, "print per-case retrieval results")
	cmd.Flags().Float64Var(&failUnderHit, "fail-under-hit", 0, "fail if aggregate hit@k is below this threshold")
	cmd.Flags().Float64Var(&failUnderRecall, "fail-under-recall", 0, "fail if aggregate recall@k is below this threshold")
	addRetrievalFlags(cmd, &retrievalFlags)
	return cmd
}

func loadRetrievalEvalDataset(path string) ([]retrievaleval.Case, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dataset: %w", err)
	}
	defer file.Close()
	cases, err := retrievaleval.LoadJSONL(file)
	if err != nil {
		return nil, err
	}
	return cases, nil
}

func validateThreshold(name string, value float64) error {
	if value < 0 || value > 1 {
		return fmt.Errorf("%s must be between 0 and 1", name)
	}
	return nil
}
