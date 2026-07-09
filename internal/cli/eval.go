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

type evalRetrievalOptions struct {
	datasetPath     string
	topK            int
	format          string
	details         bool
	failUnderHit    float64
	failUnderRecall float64
	retrievalFlags  retrievalFlagValues
}

func (e *evalRetrievalOptions) validate() error {
	if e.datasetPath == "" {
		return fmt.Errorf("--dataset is required")
	}
	if e.topK <= 0 {
		return fmt.Errorf("--top-k must be > 0")
	}
	if e.format != "text" && e.format != "json" {
		return fmt.Errorf("--format must be text or json")
	}
	if err := validateThreshold("--fail-under-hit", e.failUnderHit); err != nil {
		return err
	}
	if err := validateThreshold("--fail-under-recall", e.failUnderRecall); err != nil {
		return err
	}
	return nil
}

func newEvalRetrievalCommand(opts *options) *cobra.Command {
	var evalOpts evalRetrievalOptions
	cmd := &cobra.Command{
		Use:   "retrieval",
		Short: "Evaluate retrieval quality against a JSONL dataset",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			return runEvalRetrieval(cmd, opts, &evalOpts)
		},
	}
	cmd.Flags().StringVar(&evalOpts.datasetPath, "dataset", "", "JSONL retrieval evaluation dataset")
	cmd.Flags().IntVar(&evalOpts.topK, "top-k", 8, "number of chunks to retrieve per case")
	cmd.Flags().StringVar(&evalOpts.format, "format", "text", "output format: text or json")
	cmd.Flags().BoolVar(&evalOpts.details, "details", false, "print per-case retrieval results")
	cmd.Flags().Float64Var(&evalOpts.failUnderHit, "fail-under-hit", 0, "fail if aggregate hit@k is below this threshold")
	cmd.Flags().Float64Var(&evalOpts.failUnderRecall, "fail-under-recall", 0, "fail if aggregate recall@k is below this threshold")
	addRetrievalFlags(cmd, &evalOpts.retrievalFlags)
	return cmd
}

func runEvalRetrieval(cmd *cobra.Command, opts *options, evalOpts *evalRetrievalOptions) error {
	ctx, cancel := commandContext(cmd, opts)
	defer cancel()

	if err := evalOpts.validate(); err != nil {
		return err
	}
	retrievalOpts, err := retrievalOptionsFromFlags(cmd, evalOpts.retrievalFlags)
	if err != nil {
		return err
	}
	includeDetails := evalOpts.details || evalOpts.retrievalFlags.showQueryPlan

	paths, err := resolvePaths(opts)
	if err != nil {
		return fmt.Errorf("resolve state paths: %w", err)
	}
	if err := rag.CheckIndexForMode(paths, retrievalOpts.Mode); err != nil {
		return err
	}
	cases, err := loadRetrievalEvalDataset(evalOpts.datasetPath)
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
		TopK:              evalOpts.topK,
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
	if evalOpts.format == "json" {
		if err := retrievaleval.WriteJSON(cmd.OutOrStdout(), summary); err != nil {
			return fmt.Errorf("write retrieval evaluation JSON: %w", err)
		}
	} else {
		if err := retrievaleval.WriteText(cmd.OutOrStdout(), evalOpts.datasetPath, summary, includeDetails); err != nil {
			return fmt.Errorf("write retrieval evaluation report: %w", err)
		}
	}
	if evalOpts.failUnderHit > 0 && summary.HitAtK < evalOpts.failUnderHit {
		return fmt.Errorf("retrieval evaluation failed: hit@%d %.4f below threshold %.4f", summary.TopK, summary.HitAtK, evalOpts.failUnderHit)
	}
	if evalOpts.failUnderRecall > 0 && summary.RecallAtK < evalOpts.failUnderRecall {
		return fmt.Errorf("retrieval evaluation failed: recall@%d %.4f below threshold %.4f", summary.TopK, summary.RecallAtK, evalOpts.failUnderRecall)
	}
	return nil
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
