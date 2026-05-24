package cli

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

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
	var sourcesMode string
	var agentCoverageNudge bool
	var noAgentCoverageNudge bool
	var minAgentSearches int
	var maxAgentCoverageRetries int
	var agentReadSourceTool bool
	var noAgentReadSourceTool bool
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
			if err := validateSourcesMode(sourcesMode); err != nil {
				return err
			}
			effectiveSourcesMode := sourcesMode
			if agentMode && !cmd.Flags().Changed("sources") {
				effectiveSourcesMode = "both"
			}

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
				effectiveReadSourceTool, err := agentReadSourceToolFromFlags(agentReadSourceTool, noAgentReadSourceTool, cmd.Flags().Changed("agent-read-source-tool"), args[0])
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
					EnableReadSourceTool:  effectiveReadSourceTool,
				}
				if showAgentTools {
					toolPrinter := newAgentToolPrinter()
					agentOptions.Events = func(event falken.Event) {
						toolPrinter.printEvent(cmd.ErrOrStderr(), event)
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
				printAnswerWithSourceMode(cmd.OutOrStdout(), result.Answer, result.Sources, effectiveSourcesMode)
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
			printAnswerWithSourcesMode(cmd.OutOrStdout(), result, effectiveSourcesMode)
			return nil
		},
	}
	cmd.Flags().IntVar(&topK, "top-k", 8, "number of chunks to retrieve")
	cmd.Flags().StringVar(&model, "model", "", "LLM model override")
	cmd.Flags().IntVar(&openSource, "open-source", 0, "open retrieved source number in configured editor")
	cmd.Flags().BoolVar(&noCitationValidation, "no-citation-validation", false, "disable answer citation validation")
	cmd.Flags().BoolVar(&noCitationRetry, "no-citation-retry", false, "disable stricter citation retry")
	cmd.Flags().StringVar(&sourcesMode, "sources", "cited", "sources to print with answers: cited, all, or both")
	cmd.Flags().BoolVar(&agentMode, "agent", false, "use a Falken agent with a search_index tool instead of pre-attaching retrieved chunks")
	cmd.Flags().BoolVar(&showAgentTools, "show-agent-tools", false, "print agent tool calls and compact tool results to stderr")
	cmd.Flags().IntVar(&maxAgentSearches, "max-agent-searches", 6, "maximum number of search_index calls allowed during one agent ask run")
	cmd.Flags().BoolVar(&agentCoverageNudge, "agent-coverage-nudge", false, "enable broad-question coverage nudging in agent mode")
	cmd.Flags().BoolVar(&noAgentCoverageNudge, "no-agent-coverage-nudge", false, "disable broad-question coverage nudging in agent mode")
	cmd.Flags().IntVar(&minAgentSearches, "min-agent-searches", 0, "minimum successful search_index calls for broad agent questions")
	cmd.Flags().IntVar(&maxAgentCoverageRetries, "max-agent-coverage-retries", 0, "maximum broad-question coverage retries in agent mode")
	cmd.Flags().BoolVar(&agentReadSourceTool, "agent-read-source-tool", false, "enable the read_index_source tool in agent mode")
	cmd.Flags().BoolVar(&noAgentReadSourceTool, "no-agent-read-source-tool", false, "disable automatic read_index_source use in agent mode")
	addRetrievalFlags(cmd, &retrievalFlags)
	addSourceFilterFlags(cmd, &sourceFlags)
	return cmd
}

func printAnswer(w io.Writer, result rag.AskResult) {
	printAnswerText(w, result.Answer, result.Sources)
}

func printAnswerWithSourcesMode(w io.Writer, result rag.AskResult, sourcesMode string) {
	printAnswerWithSourceMode(w, result.Answer, result.Sources, sourcesMode)
}

func printAnswerWithSourceMode(w io.Writer, answer string, sources []rag.SourceChunk, sourcesMode string) {
	fmt.Fprintln(w, "Answer:")
	fmt.Fprintln(w)
	fmt.Fprintln(w, answer)
	fmt.Fprintln(w)
	printSourcesForMode(w, answer, sources, sourcesMode)
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

func printSourcesForMode(w io.Writer, answer string, sources []rag.SourceChunk, mode string) {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "cited":
		printSourcesWithHeading(w, "Sources:", citedSourcesOnly(answer, sources))
	case "all":
		printAllSourcesWithCitationMarkers(w, answer, sources)
		printSourceAudit(w, answer, sources)
	case "both":
		cited := citedSourcesOnly(answer, sources)
		if len(cited) != 0 {
			printSourcesWithHeading(w, "Sources cited:", cited)
		} else if len(sources) != 0 {
			fmt.Fprintln(w, "No cited sources.")
		}
		printAllSourcesWithCitationMarkers(w, answer, sources)
		printSourceAudit(w, answer, sources)
	}
}

func validateSourcesMode(mode string) error {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "", "cited", "all", "both":
		return nil
	default:
		return fmt.Errorf("--sources must be cited, all, or both")
	}
}

func printSourcesWithHeading(w io.Writer, heading string, sources []rag.SourceChunk) {
	if len(sources) == 0 {
		return
	}
	fmt.Fprintln(w, heading)
	for _, source := range sources {
		fmt.Fprintln(w, SourceReference(source))
	}
}

func printAllSourcesWithCitationMarkers(w io.Writer, answer string, sources []rag.SourceChunk) {
	if len(sources) == 0 {
		return
	}
	citedSet := citedSourceNumberSet(answer)
	fmt.Fprintln(w, "Sources available to the agent:")
	for _, source := range sources {
		ref := SourceReference(source)
		if _, ok := citedSet[source.SourceNumber]; ok {
			ref += "  (cited)"
		}
		fmt.Fprintln(w, ref)
	}
}

func printSourceAudit(w io.Writer, answer string, sources []rag.SourceChunk) {
	if len(sources) == 0 {
		return
	}
	citedSet := citedSourceNumberSet(answer)
	citedAvailable := 0
	for _, source := range sources {
		if _, ok := citedSet[source.SourceNumber]; ok {
			citedAvailable++
		}
	}
	fmt.Fprintln(w, "Source audit:")
	fmt.Fprintf(w, "- available to agent: %d\n", len(sources))
	fmt.Fprintf(w, "- cited in answer: %d\n", citedAvailable)
	fmt.Fprintf(w, "- uncited: %d\n", len(sources)-citedAvailable)
}

func citedSourcesOnly(answer string, sources []rag.SourceChunk) []rag.SourceChunk {
	keep := citedSourceNumberSet(answer)
	if len(keep) == 0 {
		return nil
	}
	out := make([]rag.SourceChunk, 0, len(keep))
	for _, source := range sources {
		if _, ok := keep[source.SourceNumber]; ok {
			out = append(out, source)
		}
	}
	return out
}

func citedSourceNumberSet(answer string) map[int]struct{} {
	cited := rag.ExtractCitedSourceNumbers(answer)
	out := make(map[int]struct{}, len(cited))
	for _, number := range cited {
		out[number] = struct{}{}
	}
	return out
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

func agentReadSourceToolFromFlags(enable, disable, enableChanged bool, question string) (bool, error) {
	if enable && disable {
		return false, errors.New("--agent-read-source-tool and --no-agent-read-source-tool cannot both be set")
	}
	if disable {
		return false, nil
	}
	if enable {
		return true, nil
	}
	if !enableChanged && agentask.IsBroadQuestion(question) {
		return true, nil
	}
	return false, nil
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
