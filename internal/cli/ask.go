package cli

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/agentask"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/spf13/cobra"
)

var newCLIChatClient = func() (llm.Client, error) {
	return llm.NewEnvChatClient(os.Getenv)
}

var askWithLLM = rag.Ask

var runAgentAsk = agentask.Run

func newAskCommand(opts *options) *cobra.Command {
	var topK int
	var model string
	var openSource int
	var noCitationValidation bool
	var noCitationRetry bool
	var agentMode bool
	var showAgentTools bool
	var agentCoverageNudge bool
	var noAgentCoverageNudge bool
	var minAgentSearches int
	var maxAgentCoverageRetries int
	var agentReadSourceTool bool
	var maxAgentSearches int
	var retrievalFlags retrievalFlagValues
	var sourceFlags sourceFilterFlagValues
	cmd := &cobra.Command{
		Use:   "ask <question>",
		Short: "Answer a question using retrieved indexed chunks",
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
			if agentMode {
				if err := checkAgentAskManifest(paths.ManifestPath); err != nil {
					return err
				}
			} else {
				if err := rag.CheckIndexForMode(paths, retrievalOpts.Mode); err != nil {
					return err
				}
			}
			store, err := manifest.Open(paths.ManifestPath)
			if err != nil {
				return fmt.Errorf("open manifest database: %w", err)
			}
			defer store.Close()
			citationPolicy := citationPolicyFromFlags(noCitationValidation, noCitationRetry)
			if agentMode {
				coverageNudge, err := agentCoverageNudgeFromFlags(agentCoverageNudge, noAgentCoverageNudge)
				if err != nil {
					return err
				}
				retrievalOpts.TopK = topK
				retrievalOpts.Paths = paths
				agentLLM, err := newCLIAgentLLMWithModel(model)
				if err != nil {
					return fmt.Errorf("configure agent LLM: %w", err)
				}
				agentOptions := agentask.Options{
					Question:              args[0],
					Paths:                 paths,
					Store:                 store,
					RetrievalDefaults:     retrievalOpts,
					AgentLLM:              agentLLM,
					EmbedderFactory:       newCLIEmbedder,
					RetrieveWithPlan:      retrieveWithPlan,
					PrepareLexicalIndex:   prepareLexicalIndex,
					ConfigureQueryPlanner: configureQueryPlanner,
					CitationPolicy:        citationPolicy,
					MaxSearchCalls:        maxAgentSearches,
					CoverageNudge:         coverageNudge,
					MinBroadSearchCalls:   minAgentSearches,
					MaxCoverageRetries:    maxAgentCoverageRetries,
					EnableReadSourceTool:  agentReadSourceTool,
				}
				if showAgentTools {
					agentOptions.Events = func(event falken.Event) {
						printAgentToolEvent(cmd.ErrOrStderr(), event)
					}
				}
				result, err := runAgentAsk(ctx, agentOptions)
				if err != nil {
					return fmt.Errorf("ask agent: %w", err)
				}
				if openSource > 0 {
					if err := openAnswerSource(cmd.ErrOrStderr(), result.Sources, openSource); err != nil {
						return err
					}
				}
				if showAgentTools {
					printCoverageWarnings(cmd.ErrOrStderr(), result.CoverageWarnings)
				}
				printCitationWarnings(cmd.ErrOrStderr(), result.CitationWarnings)
				printAnswerText(cmd.OutOrStdout(), result.Answer, result.Sources)
				return nil
			}
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
			retrievalResult, err := retrieveWithPlan(ctx, store, retrievalOpts)
			if err != nil {
				return err
			}
			chunks := retrievalResult.Chunks
			if retrievalFlags.showQueryPlan {
				printQueryPlan(cmd.OutOrStdout(), retrievalResult.Plan)
			}
			if openSource > 0 {
				if err := openRetrievedSource(cmd.ErrOrStderr(), chunks, openSource); err != nil {
					return err
				}
			}
			if len(chunks) == 0 {
				fmt.Fprintln(cmd.OutOrStdout(), "No relevant chunks found.")
				return nil
			}
			client, err := newCLIChatClient()
			if err != nil {
				return fmt.Errorf("configure LLM client: %w", err)
			}
			result, err := askWithLLM(ctx, rag.AskOptions{
				Question:       args[0],
				Model:          model,
				Chunks:         chunks,
				LLM:            client,
				CitationPolicy: citationPolicy,
			})
			if err != nil {
				printSources(cmd.OutOrStdout(), rag.SourceChunksFromRetrieved(chunks))
				return fmt.Errorf("ask LLM: %w", err)
			}
			printCitationWarnings(cmd.ErrOrStderr(), result.CitationWarnings)
			printAnswer(cmd.OutOrStdout(), result)
			return nil
		},
	}
	cmd.Flags().IntVar(&topK, "top-k", 8, "number of chunks to retrieve")
	cmd.Flags().StringVar(&model, "model", "", "LLM model override")
	cmd.Flags().IntVar(&openSource, "open-source", 0, "open retrieved source number in configured editor")
	cmd.Flags().BoolVar(&noCitationValidation, "no-citation-validation", false, "disable answer citation validation")
	cmd.Flags().BoolVar(&noCitationRetry, "no-citation-retry", false, "disable stricter citation retry")
	cmd.Flags().BoolVar(&agentMode, "agent", false, "use a Falken agent with a search_index tool instead of pre-attaching retrieved chunks")
	cmd.Flags().BoolVar(&showAgentTools, "show-agent-tools", false, "print agent tool calls and compact tool results to stderr")
	cmd.Flags().IntVar(&maxAgentSearches, "max-agent-searches", 6, "maximum number of search_index calls allowed during one agent ask run")
	cmd.Flags().BoolVar(&agentCoverageNudge, "agent-coverage-nudge", false, "enable broad-question coverage nudging in agent mode")
	cmd.Flags().BoolVar(&noAgentCoverageNudge, "no-agent-coverage-nudge", false, "disable broad-question coverage nudging in agent mode")
	cmd.Flags().IntVar(&minAgentSearches, "min-agent-searches", 0, "minimum successful search_index calls for broad agent questions")
	cmd.Flags().IntVar(&maxAgentCoverageRetries, "max-agent-coverage-retries", 0, "maximum broad-question coverage retries in agent mode")
	cmd.Flags().BoolVar(&agentReadSourceTool, "agent-read-source-tool", false, "enable the read_index_source tool in agent mode")
	addRetrievalFlags(cmd, &retrievalFlags)
	addSourceFilterFlags(cmd, &sourceFlags)
	return cmd
}

func printAnswer(w io.Writer, result rag.AskResult) {
	printAnswerText(w, result.Answer, result.Sources)
}

func printAnswerText(w io.Writer, answer string, sources []rag.SourceChunk) {
	fmt.Fprintln(w, "Answer:")
	fmt.Fprintln(w)
	fmt.Fprintln(w, answer)
	fmt.Fprintln(w)
	printSources(w, sources)
}

func printCitationWarnings(w io.Writer, warnings []string) {
	for _, warning := range warnings {
		fmt.Fprintf(w, "warning: %s\n", warning)
	}
}

func printCoverageWarnings(w io.Writer, warnings []string) {
	for _, warning := range warnings {
		fmt.Fprintf(w, "agent %s\n", warning)
	}
}

func citationPolicyFromFlags(noCitationValidation, noCitationRetry bool) rag.CitationPolicy {
	if noCitationValidation {
		return rag.CitationPolicyOff
	}
	if noCitationRetry {
		return rag.CitationPolicyValidateOnly
	}
	return rag.CitationPolicyValidateAndRetry
}

func agentCoverageNudgeFromFlags(enable, disable bool) (*bool, error) {
	if enable && disable {
		return nil, errors.New("--agent-coverage-nudge and --no-agent-coverage-nudge cannot both be set")
	}
	if enable {
		value := true
		return &value, nil
	}
	if disable {
		value := false
		return &value, nil
	}
	return nil, nil
}

func checkAgentAskManifest(manifestPath string) error {
	if _, err := os.Stat(manifestPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return rag.ErrNoIndex
		}
		return err
	}
	return nil
}
