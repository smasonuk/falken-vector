package agentask

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/rag"
)

func Run(ctx context.Context, opts Options) (final Result, err error) {
	defer func() {
		if err == nil {
			if !opts.DisableDocumentPromotion {
				final.DocumentPromotionPolicy = "auto"
			}
			populateResultToolMetrics(&final)
			if final.ReadSourceCalls > 0 {
				final.ReadSourceOverlapPolicy = string(normalizeReadSourceOverlapPolicy(opts.ReadSourceOverlapPolicy))
			}
		}
	}()
	if strings.TrimSpace(opts.Question) == "" {
		return Result{}, errors.New("question is required")
	}
	if opts.AgentLLM == nil {
		return Result{}, errors.New("agent LLM is required")
	}
	if opts.Store == nil {
		return Result{}, errors.New("manifest store is required")
	}
	if opts.Paths.ManifestPath == "" {
		return Result{}, errors.New("paths are required")
	}
	if opts.RetrieveWithPlan == nil {
		opts.RetrieveWithPlan = rag.RetrieveWithPlan
	}
	if opts.MaxSearchCalls <= 0 {
		opts.MaxSearchCalls = 6
	}
	if opts.MaxToolTopK <= 0 {
		opts.MaxToolTopK = 20
	}
	if opts.ReadSourceOverlapPolicy == ReadSourceOverlapDefault {
		opts.ReadSourceOverlapPolicy = ReadSourceOverlapSkip
	}
	coveragePolicy := normalizeCoveragePolicy(opts)
	thinPolicy := normalizeThinSourcePolicy(opts)
	documentPromotionOpts := documentPromotionOptionsFromRunOptions(opts)

	registry := NewCitationRegistry()
	searchTool := NewSearchIndexTool(SearchToolOptions{
		Paths:                 opts.Paths,
		Store:                 opts.Store,
		UserQuestion:          opts.Question,
		RetrievalDefaults:     opts.RetrievalDefaults,
		Registry:              registry,
		EmbedderFactory:       opts.EmbedderFactory,
		RetrieveWithPlan:      opts.RetrieveWithPlan,
		PrepareLexicalIndex:   opts.PrepareLexicalIndex,
		ConfigureQueryPlanner: opts.ConfigureQueryPlanner,
		MaxSearchCalls:        opts.MaxSearchCalls,
		MaxTopK:               opts.MaxToolTopK,
		MaxExpansionQueries:   opts.MaxBroadExpansionQueries,
		MaxRetrievalCalls:     opts.MaxRetrievalCalls,
		DocumentPromotion:     documentPromotionOpts,
	})
	agentTools := []falken.Tool{searchTool}
	if opts.EnableReadSourceTool {
		agentTools = append(agentTools, NewReadIndexSourceTool(ReadSourceToolOptions{
			Registry:                 registry,
			OverlapPolicy:            opts.ReadSourceOverlapPolicy,
			MaxMergedReadSourceLines: opts.MaxMergedReadSourceLines,
		}))
	}
	if !opts.DisableDocumentPromotion {
		agentTools = append(agentTools, NewReadIndexDocumentTool(ReadDocumentToolOptions{
			Registry:  registry,
			MaxLines:  documentPromotionOpts.MaxDocumentReadLines,
			MaxTokens: documentPromotionOpts.MaxDocumentReadTokens,
		}))
	}

	var capturedToolCalls []string
	trace := AgentTrace{}
	events := func(event falken.Event) {
		if event.ToolCall != nil {
			capturedToolCalls = append(capturedToolCalls, event.ToolCall.Name)
			trace.ToolCalls = append(trace.ToolCalls, toolCallRecord(*event.ToolCall))
		}
		if event.ToolResult != nil {
			trace.ToolResults = append(trace.ToolResults, toolResultRecord(*event.ToolResult))
		}
		if opts.Events != nil {
			opts.Events(event)
		}
	}

	agent, err := falken.NewAgent(ctx, falken.AgentConfig{
		LLM:          opts.AgentLLM,
		SystemPrompt: agentSystemPrompt(opts.EnableReadSourceTool, !opts.DisableDocumentPromotion),
		Tools:        agentTools,
		Events:       events,
		Permissions: falken.SimplePermissions{
			AllowNetwork: true,
		},
	})
	if err != nil {
		return Result{}, err
	}
	defer agent.Close(context.Background())

	answer, err := agent.Run(ctx, opts.Question)
	if err != nil {
		return Result{}, err
	}

	result := Result{
		Answer:    answer,
		Sources:   registry.Sources(),
		ToolCalls: append([]string(nil), capturedToolCalls...),
		Trace:     cloneAgentTrace(trace),
	}
	normalizeResultCitations(&result)
	answer = result.Answer
	policy := normalizeAgentCitationPolicy(opts.CitationPolicy)
	if policy == rag.CitationPolicyOff {
		result.CitationValid = true
		return result, nil
	}
	validation := registry.Validate(answer)
	if validation.Valid && len(result.Sources) == 0 && len(result.ToolCalls) == 0 {
		result.CitationWarnings = append(result.CitationWarnings, "agent did not return cited indexed sources")
		result.CitationValid = false
		return result, nil
	}
	if unsupportedZeroSourceAnswer(validation, result.Sources, result.ToolCalls, answer) {
		result.CitationWarnings = append(result.CitationWarnings, unsupportedZeroSourceAnswerWarning)
		if policy != rag.CitationPolicyValidateAndRetry {
			result.CitationValid = false
			return result, nil
		}
		retryAnswer, err := agent.Run(ctx, correctiveUnsupportedAnswerPrompt(answer))
		if err != nil {
			result.CitationWarnings = append(result.CitationWarnings, "citation retry failed: "+err.Error())
			return result, nil
		}
		result.Retried = true
		result.Answer = retryAnswer
		normalizeResultCitations(&result)
		result.Sources = registry.Sources()
		result.ToolCalls = append([]string(nil), capturedToolCalls...)
		result.Trace = cloneAgentTrace(trace)
		return validateRetryResult(&result, registry)
	}
	if validation.Valid {
		coverageRetries := 0
		coverageComplete := false
		thinRetries := 0
		for {
			if !coverageComplete {
				stats := searchStats(trace)
				decision := coverageNudgeDecision(opts.Question, coveragePolicy, stats.SearchCalls, stats.SuccessfulSearchCalls, len(result.Sources), opts.MaxSearchCalls, coverageRetries)
				if decision.Nudge {
					result.CoverageWarnings = append(result.CoverageWarnings, coverageNudgeWarning(stats.SuccessfulSearchCalls, decision.RemainingNeeded))
					retryAnswer, err := agent.Run(ctx, coverageNudgePrompt(opts.Question, answer, stats.SuccessfulSearchCalls, decision.RemainingNeeded))
					if err != nil {
						result.CitationWarnings = append(result.CitationWarnings, "coverage nudge failed: "+err.Error())
						result.CitationValid = true
						return result, nil
					}
					coverageRetries++
					result.Retried = true
					result.CoverageNudged = true
					result.Answer = retryAnswer
					normalizeResultCitations(&result)
					result.Sources = registry.Sources()
					result.ToolCalls = append([]string(nil), capturedToolCalls...)
					result.Trace = cloneAgentTrace(trace)
					validation = registry.Validate(result.Answer)
					if !validation.Valid || unsupportedZeroSourceAnswer(validation, result.Sources, result.ToolCalls, result.Answer) {
						return validateRetryResult(&result, registry)
					}
					answer = result.Answer
					continue
				}
				if warning := coverageSkippedWarning(decision.Reason); warning != "" {
					result.CoverageWarnings = append(result.CoverageWarnings, warning)
				}
				coverageComplete = true
			}

			thinDecision := thinSourceNudgeDecision(opts.Question, opts.EnableReadSourceTool, thinPolicy, result.Answer, result.Sources, trace, thinRetries)
			if !thinDecision.Nudge {
				if thinRetries == 0 {
					if warning := thinSourceSkippedWarning(thinDecision.Reason); warning != "" {
						result.ThinSourceWarnings = append(result.ThinSourceWarnings, warning)
					}
				}
				result.CitationValid = true
				return result, nil
			}
			result.ThinSourceWarnings = append(result.ThinSourceWarnings, thinSourceNudgeWarning(thinDecision.SourceNumbers))
			emitAgentNote(opts.Events, "agent "+thinSourceNudgeWarning(thinDecision.SourceNumbers))
			retryAnswer, err := agent.Run(ctx, thinSourceNudgePrompt(opts.Question, result.Answer, thinDecision.SourceNumbers, thinDecision.ContextLines))
			if err != nil {
				result.ThinSourceWarnings = append(result.ThinSourceWarnings, "thin-source nudge failed: "+err.Error())
				result.CitationValid = true
				return result, nil
			}
			thinRetries++
			result.Retried = true
			result.ThinSourceNudged = true
			result.Answer = retryAnswer
			normalizeResultCitations(&result)
			result.Sources = registry.Sources()
			result.ToolCalls = append([]string(nil), capturedToolCalls...)
			result.Trace = cloneAgentTrace(trace)
			validation = registry.Validate(result.Answer)
			if !validation.Valid || unsupportedZeroSourceAnswer(validation, result.Sources, result.ToolCalls, result.Answer) {
				return validateRetryResult(&result, registry)
			}
			answer = result.Answer
		}
	}

	result.CitationWarnings = append(result.CitationWarnings, validation.Warnings...)
	if policy != rag.CitationPolicyValidateAndRetry {
		return result, nil
	}

	retryAnswer, err := agent.Run(ctx, correctiveCitationPrompt(validation.Warnings, answer))
	if err != nil {
		result.CitationWarnings = append(result.CitationWarnings, "citation retry failed: "+err.Error())
		return result, nil
	}
	result.Retried = true
	result.Answer = retryAnswer
	normalizeResultCitations(&result)
	result.Sources = registry.Sources()
	result.ToolCalls = append([]string(nil), capturedToolCalls...)
	result.Trace = cloneAgentTrace(trace)
	return validateRetryResult(&result, registry)
}

func validateRetryResult(result *Result, registry *CitationRegistry) (Result, error) {
	normalizeResultCitations(result)
	retryValidation := registry.Validate(result.Answer)
	if unsupportedZeroSourceAnswer(retryValidation, result.Sources, result.ToolCalls, result.Answer) {
		result.CitationValid = false
		result.CitationWarnings = append(result.CitationWarnings, "final answer may have citation issues: "+unsupportedZeroSourceAnswerWarning)
		return *result, nil
	}
	result.CitationValid = retryValidation.Valid
	if retryValidation.Valid && len(result.Sources) == 0 && len(result.ToolCalls) == 0 {
		result.CitationValid = false
		result.CitationWarnings = append(result.CitationWarnings, "agent did not return cited indexed sources")
		return *result, nil
	}
	if !retryValidation.Valid {
		result.CitationWarnings = append(result.CitationWarnings, "final answer may have citation issues: "+strings.Join(retryValidation.Warnings, "; "))
	}
	return *result, nil
}

func normalizeResultCitations(result *Result) {
	normalized, notes := rag.NormalizeGroupedCitations(result.Answer)
	result.Answer = normalized
	result.CitationNotes = appendUniqueStrings(result.CitationNotes, notes...)
}

func populateResultToolMetrics(result *Result) {
	if result == nil {
		return
	}
	searchCalls, retrievalCalls, readSourceStats, readDocumentStats := agentToolMetrics(*result)
	result.SearchToolCalls = searchCalls
	result.RetrievalCalls = retrievalCalls
	result.ReadSourceCalls = readSourceStats.Calls
	result.ReadSourceAlreadyCovered = readSourceStats.AlreadyCovered
	result.ReadSourceMerges = readSourceStats.Merges
	result.ReadSourceMergeTooLarge = readSourceStats.MergeTooLarge
	result.DocumentReadCalls = readDocumentStats.Calls + readDocumentStats.AutoReads
	result.DocumentReadWholeCalls = readDocumentStats.WholeCalls + readDocumentStats.AutoReads
	result.DocumentReadRangeCalls = readDocumentStats.RangeCalls
	result.DocumentReadParentCalls = readDocumentStats.ParentCalls
	result.DocumentPromotionAutoReads = readDocumentStats.AutoReads
	result.DocumentPromotionSkippedTooLarge = readDocumentStats.SkippedTooLarge
	result.DocumentPromotionSkippedBudget = readDocumentStats.SkippedBudget
	if readDocumentStats.Calls > 0 || readDocumentStats.AutoReads > 0 || result.DocumentPromotionPolicy != "" {
		result.DocumentPromotionPolicy = "auto"
	}
}

type readSourceMetricStats struct {
	Calls          int
	AlreadyCovered int
	Merges         int
	MergeTooLarge  int
}

type readDocumentMetricStats struct {
	Calls           int
	WholeCalls      int
	RangeCalls      int
	ParentCalls     int
	AutoReads       int
	SkippedTooLarge int
	SkippedBudget   int
}

func agentToolMetrics(result Result) (int, int, readSourceMetricStats, readDocumentMetricStats) {
	searchCalls := 0
	readStats := readSourceMetricStats{}
	documentStats := readDocumentMetricStats{}
	if len(result.Trace.ToolCalls) != 0 {
		for _, call := range result.Trace.ToolCalls {
			if call.Name == SearchIndexToolName {
				searchCalls++
			}
			if call.Name == ReadIndexSourceToolName {
				readStats.Calls++
			}
			if call.Name == ReadIndexDocumentToolName {
				documentStats.Calls++
			}
		}
	} else {
		for _, name := range result.ToolCalls {
			if name == SearchIndexToolName {
				searchCalls++
			}
			if name == ReadIndexSourceToolName {
				readStats.Calls++
			}
			if name == ReadIndexDocumentToolName {
				documentStats.Calls++
			}
		}
	}
	retrievalCalls := 0
	for _, toolResult := range result.Trace.ToolResults {
		if len(toolResult.Payload) == 0 {
			continue
		}
		switch toolResult.Name {
		case SearchIndexToolName:
			var payload struct {
				RetrievalCalls     int                         `json:"retrieval_calls"`
				DocumentPromotions []DocumentPromotionDecision `json:"document_promotions"`
			}
			if json.Unmarshal(toolResult.Payload, &payload) == nil {
				retrievalCalls += payload.RetrievalCalls
				for _, decision := range payload.DocumentPromotions {
					switch decision.Status {
					case "ok":
						documentStats.AutoReads++
					case "too_large":
						documentStats.SkippedTooLarge++
					case "budget_exhausted":
						documentStats.SkippedBudget++
					}
				}
			}
		case ReadIndexSourceToolName:
			var payload struct {
				Status string `json:"status"`
			}
			if json.Unmarshal(toolResult.Payload, &payload) != nil {
				continue
			}
			switch payload.Status {
			case "already_covered":
				readStats.AlreadyCovered++
			case "merge_existing":
				readStats.Merges++
			case "merge_too_large":
				readStats.MergeTooLarge++
			}
		case ReadIndexDocumentToolName:
			var payload struct {
				Status string `json:"status"`
				Mode   string `json:"mode"`
			}
			if json.Unmarshal(toolResult.Payload, &payload) != nil {
				continue
			}
			if payload.Status == "too_large" {
				documentStats.SkippedTooLarge++
			}
			switch payload.Mode {
			case "range":
				documentStats.RangeCalls++
			case "parent":
				documentStats.ParentCalls++
			default:
				documentStats.WholeCalls++
			}
		}
	}
	return searchCalls, retrievalCalls, readStats, documentStats
}

func documentPromotionOptionsFromRunOptions(opts Options) DocumentPromotionOptions {
	promotion := defaultDocumentPromotionOptions()
	promotion.Enabled = !opts.DisableDocumentPromotion
	if opts.MaxDocumentReadLines > 0 {
		promotion.MaxDocumentReadLines = opts.MaxDocumentReadLines
	}
	if opts.MaxDocumentReadTokens > 0 {
		promotion.MaxDocumentReadTokens = opts.MaxDocumentReadTokens
	}
	if opts.MaxDocumentReads > 0 {
		promotion.MaxDocumentReads = opts.MaxDocumentReads
	}
	if opts.SmallDocumentLineLimit > 0 {
		promotion.SmallFileLineLimit = opts.SmallDocumentLineLimit
	}
	if opts.DocumentDominantShare > 0 {
		promotion.DominantDocShare = opts.DocumentDominantShare
	}
	if opts.DocumentMinHits > 0 {
		promotion.MinHitsForDominantDoc = opts.DocumentMinHits
	}
	return promotion
}

func appendUniqueStrings(values []string, additions ...string) []string {
	seen := make(map[string]struct{}, len(values)+len(additions))
	for _, value := range values {
		seen[value] = struct{}{}
	}
	for _, value := range additions {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		values = append(values, value)
	}
	return values
}

func correctiveCitationPrompt(warnings []string, previousAnswer string) string {
	return fmt.Sprintf(`Your previous answer had citation issues:

%s

Rewrite the answer using only source IDs returned by search_index.
Every factual claim about indexed content must cite an existing [source N].
Use one bracket per cited source: [source 2] [source 3].
Never write [sources 2, 3], [source 2, 3], [source 2 and 3], or multiple source numbers inside one bracket.
Do not cite source IDs that were not returned.
If more evidence is needed, call search_index again.
If the indexed corpus does not support the answer, say you do not know.

Previous answer:
%s`, strings.Join(warnings, "\n"), previousAnswer)
}

func correctiveUnsupportedAnswerPrompt(previousAnswer string) string {
	return fmt.Sprintf(`Your previous answer was unsupported.

search_index returned no source chunks, so you must not answer factual questions about indexed files, code, documentation, or project behaviour from memory.

Either:
1. call search_index again with a better query, or
2. say that the indexed corpus does not contain enough information to answer.

Do not invent paths, line numbers, file contents, APIs, or source IDs.
Do not cite [source N] unless search_index returned that source.
Use one bracket per cited source: [source 2] [source 3].
Never write [sources 2, 3], [source 2, 3], [source 2 and 3], or multiple source numbers inside one bracket.

Previous answer:
%s`, previousAnswer)
}

const unsupportedZeroSourceAnswerWarning = "agent returned an unsupported answer after search_index returned no sources"

func unsupportedZeroSourceAnswer(validation rag.CitationValidation, sources []rag.SourceChunk, toolCalls []string, answer string) bool {
	return validation.Valid &&
		len(sources) == 0 &&
		hasToolCall(toolCalls, SearchIndexToolName) &&
		!answerAcknowledgesNoSupport(answer)
}

func hasToolCall(toolCalls []string, name string) bool {
	for _, call := range toolCalls {
		if call == name {
			return true
		}
	}
	return false
}

func answerAcknowledgesNoSupport(answer string) bool {
	normalized := strings.ToLower(strings.TrimSpace(answer))
	if normalized == "" {
		return true
	}
	phrases := []string{
		"i don't know",
		"i do not know",
		"not enough information",
		"not enough evidence",
		"could not find",
		"can't find",
		"cannot find",
		"not found",
		"no relevant",
		"no supporting",
		"does not contain",
		"doesn't contain",
		"indexed corpus does not contain",
		"search_index did not return",
		"search_index returned no",
	}
	for _, phrase := range phrases {
		if strings.Contains(normalized, phrase) {
			return true
		}
	}
	return false
}

func emitAgentNote(events falken.EventSink, text string) {
	text = strings.TrimSpace(text)
	if events == nil || text == "" {
		return
	}
	events(falken.Event{Type: falken.EventThought, Text: text})
}

func normalizeAgentCitationPolicy(policy rag.CitationPolicy) rag.CitationPolicy {
	switch policy {
	case rag.CitationPolicyValidateOnly, rag.CitationPolicyOff:
		return policy
	case "", rag.CitationPolicyValidateAndRetry:
		return rag.CitationPolicyValidateAndRetry
	default:
		return rag.CitationPolicyValidateAndRetry
	}
}
