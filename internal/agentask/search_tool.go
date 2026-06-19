package agentask

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
)

const SearchIndexToolName = "search_index"

type SearchToolOptions struct {
	Paths config.Paths
	Store manifest.Store

	UserQuestion string

	RetrievalDefaults rag.RetrieveOptions

	Registry *CitationRegistry

	EmbedderFactory func() (llm.Embedder, error)

	RetrieveWithPlan func(
		context.Context,
		manifest.Store,
		rag.RetrieveOptions,
	) (rag.RetrieveResult, error)

	PrepareLexicalIndex func(
		context.Context,
		manifest.Store,
		rag.RetrievalMode,
	) error

	ConfigureQueryPlanner func(*rag.RetrieveOptions) error

	MaxSearchCalls      int
	MaxTopK             int
	MaxExpansionQueries int
	MaxRetrievalCalls   int

	DocumentPromotion DocumentPromotionOptions
	ReadFile          func(string) ([]byte, error)
}

type searchIndexToolState struct {
	mu                      sync.Mutex
	calls                   int
	retrievalCalls          int
	documentPromotionReads  int
	documentPromotionTokens int
	promotedDocuments       map[string]struct{}
	documentMatchHits       map[string]int
	opts                    SearchToolOptions
}

type searchIndexArgs struct {
	Query     string `json:"query"`
	TopK      *int   `json:"top_k"`
	Retrieval string `json:"retrieval"`
	Strategy  string `json:"strategy"`
}

func NewSearchIndexTool(opts SearchToolOptions) falken.Tool {
	if opts.Registry == nil {
		opts.Registry = NewCitationRegistry()
	}
	state := &searchIndexToolState{opts: opts}
	return falken.ToolFunc(searchIndexDescriptor(), state.execute)
}

func searchIndexDescriptor() falken.ToolDescriptor {
	return falken.ToolDescriptor{
		Name:        SearchIndexToolName,
		Description: "Search the local Falken vector/lexical index and return cited source snippets. Use this before answering questions about indexed files, code, documentation, or project facts.",
		Parameters: json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "required": ["query"],
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query for the local indexed corpus."
    },
    "top_k": {
      "type": "integer",
      "description": "Maximum number of source chunks to return. Defaults to the CLI --top-k value and is capped by the tool."
    },
    "retrieval": {
      "type": "string",
      "enum": ["vector", "lexical", "hybrid"],
      "description": "Optional retrieval mode override for this search. Defaults to the CLI --retrieval value."
    },
    "strategy": {
      "type": "string",
      "enum": ["focused", "broad"],
      "description": "Search strategy. focused runs the requested query. broad expands from first-pass source terms and fuses follow-up results."
    }
  }
}`),
		Safety: falken.ToolSafety{
			ReadsHostState: true,
			ReadsWorkspace: true,
			UsesNetwork:    true,
		},
	}
}

func (s *searchIndexToolState) execute(ctx context.Context, invocation falken.ToolInvocation) (falken.ToolExecutionResult, error) {
	maxSearchCalls := s.opts.MaxSearchCalls
	if maxSearchCalls <= 0 {
		maxSearchCalls = 6
	}
	args, err := decodeSearchIndexArgs(invocation.Arguments)
	if err != nil {
		return failedSearchToolResult("invalid_arguments", err.Error(), nil), nil
	}
	query := normalizeSearchQuery(args.Query)
	if query == "" {
		return failedSearchToolResult("invalid_arguments", "query is required", nil), nil
	}
	originalQuery := strings.TrimSpace(args.Query)
	queryNormalized := originalQuery != "" && query != originalQuery
	payloadOriginalQuery := ""
	if queryNormalized {
		payloadOriginalQuery = originalQuery
	}

	retrieveOpts := s.opts.RetrievalDefaults
	topK, warnings := normalizeSearchTopK(args.TopK, retrieveOpts.TopK, s.opts.MaxTopK)
	mode, err := normalizeSearchMode(args.Retrieval, retrieveOpts.Mode)
	if err != nil {
		return failedSearchToolResult("invalid_retrieval", err.Error(), nil), nil
	}
	strategy, err := normalizeSearchStrategy(args.Strategy)
	if err != nil {
		return failedSearchToolResult("invalid_strategy", err.Error(), nil), nil
	}
	if strings.TrimSpace(args.Strategy) == "" && isBroadCoverageQuestion(s.opts.UserQuestion) {
		strategy = "broad"
	}
	maxExpansionQueries := s.opts.MaxExpansionQueries
	if maxExpansionQueries < 0 {
		maxExpansionQueries = 0
	} else if maxExpansionQueries == 0 {
		maxExpansionQueries = 2
	}

	s.mu.Lock()
	if s.calls >= maxSearchCalls {
		s.mu.Unlock()
		return failedSearchToolResult("search_call_limit", "search_index call limit exceeded", nil), nil
	}
	s.calls++
	s.mu.Unlock()

	if !s.reserveRetrievalCall() {
		return failedSearchToolResult("retrieval_call_limit", "search_index retrieval call limit exceeded", nil), nil
	}
	retrieveOpts, result, failure, ok := s.retrieveOnce(ctx, retrieveOpts, query, topK, mode, true)
	if !ok {
		return failure, nil
	}
	seedPlan := result.Plan
	internalPlans := []searchQueryPlanPayload{queryPlanPayload(seedPlan)}
	retrievalCalls := 1
	expansionQueries := []string{}
	if strategy == "broad" && len(result.Chunks) != 0 && maxExpansionQueries > 0 {
		seedSources := sourceChunksFromRetrieved(result.Chunks)
		candidateExpansionQueries := SuggestBroadExpansionQueries(query, seedSources, maxExpansionQueries)
		numCandidates := len(candidateExpansionQueries)
		if numCandidates != 0 {
			expansionQueriesCap := len(expansionQueries) + numCandidates
			if cap(expansionQueries) < expansionQueriesCap {
				tmp := make([]string, len(expansionQueries), expansionQueriesCap)
				copy(tmp, expansionQueries)
				expansionQueries = tmp
			}

			internalPlansCap := len(internalPlans) + numCandidates
			if cap(internalPlans) < internalPlansCap {
				tmp := make([]searchQueryPlanPayload, len(internalPlans), internalPlansCap)
				copy(tmp, internalPlans)
				internalPlans = tmp
			}

			resultsCap := 1 + numCandidates
			results := make([]rag.RetrieveResult, 1, resultsCap)
			results[0] = result

			for _, expansionQuery := range candidateExpansionQueries {
				if !s.reserveRetrievalCall() {
					warnings = append(warnings, fmt.Sprintf("broad expansion stopped after %d expansion queries because max retrieval calls was reached", len(expansionQueries)))
					break
				}
				expansionQueries = append(expansionQueries, expansionQuery)
				expansionOpts := s.opts.RetrievalDefaults
				expansionOpts.QueryPlannerMode = rag.QueryPlannerModeNone
				expansionOpts.QueryPlanner = nil
				_, expansionResult, failure, ok := s.retrieveOnce(ctx, expansionOpts, expansionQuery, topK, mode, false)
				if !ok {
					return failure, nil
				}
				retrievalCalls++
				internalPlans = append(internalPlans, queryPlanPayload(expansionResult.Plan))
				results = append(results, expansionResult)
			}
			result = mergeBroadSearchResultsForQuery(query, results, retrieveOpts.TopK)
		}
	}
	suggestedQueries := []string{}
	if strategy == "focused" && len(result.Chunks) != 0 {
		suggestedQueries = SuggestAgentFollowupHints(query, sourceChunksFromRetrieved(result.Chunks), 3)
	}

	sources := make([]searchSourcePayload, 0, len(result.Chunks))
	registeredSources := make([]rag.SourceChunk, 0, len(result.Chunks))
	beforeSourceCount := s.opts.Registry.SourceCount()
	newSources := 0
	documentIDs := map[string]struct{}{}
	seenSourceNumbers := map[int]struct{}{}
	for i, chunk := range result.Chunks {
		source := s.opts.Registry.RegisterWithProvenance(chunk, &rag.SourceProvenance{
			ToolName:      SearchIndexToolName,
			Query:         query,
			Strategy:      strategy,
			Rank:          i + 1,
			RetrievalCall: retrievalCalls,
		})
		if source.SourceNumber > beforeSourceCount {
			if _, alreadySeen := seenSourceNumbers[source.SourceNumber]; !alreadySeen {
				newSources++
			}
		}
		seenSourceNumbers[source.SourceNumber] = struct{}{}
		registeredSources = append(registeredSources, source)
		if chunk.Chunk.DocumentID != "" {
			documentIDs[chunk.Chunk.DocumentID] = struct{}{}
		}
		sources = append(sources, searchSourcePayload{
			SourceNumber: source.SourceNumber,
			Path:         rag.DisplayPath(source.Path, source.SourceRoot),
			StartLine:    source.StartLine,
			EndLine:      source.EndLine,
			Score:        source.Score,
			ChunkID:      chunk.Chunk.ID,
			DocumentID:   chunk.Chunk.DocumentID,
			Text:         source.Text,
		})
	}
	documentMatches := BuildDocumentMatchSummaries(registeredSources, beforeSourceCount, retrieveOpts.TopK)
	promotionDecisions, promotedContents := s.applyDocumentPromotions(query, documentMatches)

	payload := searchToolPayload{
		Success:            true,
		Status:             "ok",
		Query:              query,
		OriginalQuery:      payloadOriginalQuery,
		QueryNormalized:    queryNormalized,
		Strategy:           strategy,
		Retrieval:          string(retrieveOpts.Mode),
		TopK:               retrieveOpts.TopK,
		QueryPlan:          queryPlanPayload(seedPlan),
		SeedQueryPlan:      queryPlanPayload(seedPlan),
		InternalPlans:      internalPlans,
		ExpansionQueries:   expansionQueries,
		SuggestedQueries:   suggestedQueries,
		NewSources:         newSources,
		DuplicateSources:   len(sources) - newSources,
		UniqueDocuments:    len(documentIDs),
		RetrievalCalls:     retrievalCalls,
		Sources:            sources,
		DocumentMatches:    documentMatches,
		DocumentPromotions: promotionDecisions,
		Warnings:           warnings,
	}
	return successfulSearchToolResult(renderSearchToolContent(query, strategy, seedPlan, expansionQueries, suggestedQueries, sources, warnings, payload.NewSources, payload.DuplicateSources, documentMatches, promotedContents), payload), nil
}

func (s *searchIndexToolState) applyDocumentPromotions(query string, matches []DocumentMatchSummary) ([]DocumentPromotionDecision, []string) {
	opts := normalizeDocumentPromotionOptions(s.opts.DocumentPromotion)
	if !opts.Enabled {
		return nil, nil
	}
	s.mu.Lock()
	if s.documentMatchHits == nil {
		s.documentMatchHits = map[string]int{}
	}
	remainingReads := opts.MaxDocumentReads - s.documentPromotionReads
	remainingTokens := opts.MaxTotalDocumentTokens - s.documentPromotionTokens
	promotedDocuments := make(map[string]struct{}, len(s.promotedDocuments))
	for path := range s.promotedDocuments {
		promotedDocuments[path] = struct{}{}
	}
	cumulativeMatches := make([]DocumentMatchSummary, 0, len(matches))
	for _, match := range matches {
		match.HitCount += s.documentMatchHits[match.Path]
		s.documentMatchHits[match.Path] = match.HitCount
		cumulativeMatches = append(cumulativeMatches, match)
	}
	s.mu.Unlock()
	if remainingReads <= 0 || remainingTokens <= 0 {
		return documentPromotionBudgetDecisions(matches, remainingReads, remainingTokens), nil
	}
	opts.MaxDocumentReads = remainingReads
	opts.MaxTotalDocumentTokens = remainingTokens
	matches = unpromotedDocumentMatches(cumulativeMatches, promotedDocuments)
	promotions := SelectDocumentPromotions(s.opts.UserQuestion, matches, opts)
	if len(promotions) == 0 {
		return nil, nil
	}
	readOpts := ReadDocumentToolOptions{
		Registry:  s.opts.Registry,
		ReadFile:  s.opts.ReadFile,
		MaxLines:  opts.MaxDocumentReadLines,
		MaxTokens: opts.MaxDocumentReadTokens,
	}
	decisions := make([]DocumentPromotionDecision, 0, len(promotions))
	contents := make([]string, 0, len(promotions))
	for _, promotion := range promotions {
		result := ExecuteReadIndexDocument(readDocumentRequest{
			SourceNumber: promotion.SourceNumber,
			Mode:         promotion.Mode,
			MaxLines:     opts.MaxDocumentReadLines,
			MaxTokens:    opts.MaxDocumentReadTokens,
		}, readOpts)
		decision := DocumentPromotionDecision{
			SourceNumber: promotion.SourceNumber,
			Path:         promotion.Path,
			Status:       result.Status,
			Reason:       promotion.Reason,
			MaxLines:     opts.MaxDocumentReadLines,
			MaxTokens:    opts.MaxDocumentReadTokens,
		}
		var payload readDocumentPayload
		if len(result.Payload) != 0 && json.Unmarshal(result.Payload, &payload) == nil {
			decision.StartLine = payload.StartLine
			decision.EndLine = payload.EndLine
			decision.Lines = payload.Lines
			decision.EstimatedTokens = payload.EstimatedTokens
		}
		if result.Success && result.Status == "ok" {
			contents = append(contents, result.Content)
			s.mu.Lock()
			if s.promotedDocuments == nil {
				s.promotedDocuments = map[string]struct{}{}
			}
			s.promotedDocuments[promotion.Path] = struct{}{}
			s.documentPromotionReads++
			s.documentPromotionTokens += decision.EstimatedTokens
			s.mu.Unlock()
		}
		if result.Status == "" && result.Error != "" {
			decision.Status = "read_document_failed"
		}
		if decision.Status == "" {
			decision.Status = result.Status
		}
		if strings.TrimSpace(query) != "" && decision.Reason != "" {
			decision.Reason = decision.Reason + fmt.Sprintf("; introduced by search query %q", query)
		}
		decisions = append(decisions, decision)
	}
	return decisions, contents
}

func documentPromotionBudgetDecisions(matches []DocumentMatchSummary, remainingReads int, remainingTokens int) []DocumentPromotionDecision {
	if len(matches) == 0 {
		return nil
	}
	reason := "automatic document promotion budget exhausted"
	if remainingTokens <= 0 {
		reason = "automatic document promotion token budget exhausted"
	}
	decision := DocumentPromotionDecision{
		Path:   matches[0].Path,
		Status: "budget_exhausted",
		Reason: reason,
	}
	if len(matches[0].SourceNumbers) != 0 {
		decision.SourceNumber = matches[0].SourceNumbers[0]
	}
	return []DocumentPromotionDecision{decision}
}

func unpromotedDocumentMatches(matches []DocumentMatchSummary, promoted map[string]struct{}) []DocumentMatchSummary {
	if len(matches) == 0 || len(promoted) == 0 {
		return matches
	}
	out := make([]DocumentMatchSummary, 0, len(matches))
	for _, match := range matches {
		if _, ok := promoted[match.Path]; ok {
			continue
		}
		out = append(out, match)
	}
	return out
}

func (s *searchIndexToolState) reserveRetrievalCall() bool {
	maxRetrievalCalls := s.opts.MaxRetrievalCalls
	if maxRetrievalCalls <= 0 {
		return true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.retrievalCalls >= maxRetrievalCalls {
		return false
	}
	s.retrievalCalls++
	return true
}

func decodeSearchIndexArgs(raw json.RawMessage) (searchIndexArgs, error) {
	var args searchIndexArgs
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&args); err != nil {
		return args, fmt.Errorf("decode search_index arguments: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return args, fmt.Errorf("decode search_index arguments: multiple JSON values")
	}
	return args, nil
}

func normalizeSearchTopK(value *int, defaultTopK, maxTopK int) (int, []string) {
	topK := 0
	if value != nil {
		topK = *value
	}
	warnings := []string{}
	if topK <= 0 {
		topK = defaultTopK
	}
	if value != nil && *value > 0 && defaultTopK > 0 && topK < defaultTopK {
		topK = defaultTopK
		warnings = append(warnings, fmt.Sprintf("top_k raised to configured floor %d", defaultTopK))
	}
	if topK <= 0 {
		topK = 8
	}
	if maxTopK <= 0 {
		maxTopK = 20
	}
	if topK > maxTopK {
		warnings = append(warnings, fmt.Sprintf("top_k capped at %d", maxTopK))
		return maxTopK, warnings
	}
	return topK, warnings
}

func normalizeSearchMode(value string, defaultMode rag.RetrievalMode) (rag.RetrievalMode, error) {
	if strings.TrimSpace(value) == "" {
		return rag.ParseRetrievalMode(string(defaultMode))
	}
	return rag.ParseRetrievalMode(value)
}

func normalizeSearchStrategy(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "focused", nil
	}
	switch value {
	case "focused", "broad":
		return value, nil
	default:
		return "", fmt.Errorf("invalid search strategy %q", value)
	}
}

func (s *searchIndexToolState) retrieveOnce(ctx context.Context, retrieveOpts rag.RetrieveOptions, query string, topK int, mode rag.RetrievalMode, configurePlanner bool) (rag.RetrieveOptions, rag.RetrieveResult, falken.ToolExecutionResult, bool) {
	retrieveOpts.Question = query
	retrieveOpts.TopK = topK
	retrieveOpts.Mode = mode
	retrieveOpts.Paths = s.opts.Paths

	if err := rag.CheckIndexForMode(retrieveOpts.Paths, retrieveOpts.Mode); err != nil {
		return rag.RetrieveOptions{}, rag.RetrieveResult{}, failedSearchToolResult("index_unavailable", err.Error(), nil), false
	}
	if rag.RetrievalModeUsesLexical(retrieveOpts.Mode) && s.opts.PrepareLexicalIndex != nil {
		if err := s.opts.PrepareLexicalIndex(ctx, s.opts.Store, retrieveOpts.Mode); err != nil {
			return rag.RetrieveOptions{}, rag.RetrieveResult{}, failedSearchToolResult("prepare_lexical_index_failed", err.Error(), nil), false
		}
	}
	if rag.RetrievalModeUsesVector(retrieveOpts.Mode) {
		if s.opts.EmbedderFactory == nil {
			return rag.RetrieveOptions{}, rag.RetrieveResult{}, failedSearchToolResult("missing_embedder_factory", "embedder factory is required for vector retrieval", nil), false
		}
		embedder, err := s.opts.EmbedderFactory()
		if err != nil {
			return rag.RetrieveOptions{}, rag.RetrieveResult{}, failedSearchToolResult("configure_embedder_failed", err.Error(), nil), false
		}
		retrieveOpts.Embedder = embedder
	}
	if configurePlanner && s.opts.ConfigureQueryPlanner != nil {
		if err := s.opts.ConfigureQueryPlanner(&retrieveOpts); err != nil {
			return rag.RetrieveOptions{}, rag.RetrieveResult{}, failedSearchToolResult("configure_query_planner_failed", err.Error(), nil), false
		}
	}
	normalized, err := rag.NormalizeRetrieveOptions(retrieveOpts)
	if err != nil {
		return rag.RetrieveOptions{}, rag.RetrieveResult{}, failedSearchToolResult("invalid_options", err.Error(), nil), false
	}
	retrieveWithPlan := s.opts.RetrieveWithPlan
	if retrieveWithPlan == nil {
		retrieveWithPlan = rag.RetrieveWithPlan
	}
	result, err := retrieveWithPlan(ctx, s.opts.Store, normalized)
	if err != nil {
		return rag.RetrieveOptions{}, rag.RetrieveResult{}, failedSearchToolResult("retrieval_failed", err.Error(), nil), false
	}
	return normalized, result, falken.ToolExecutionResult{}, true
}

func sourceChunksFromRetrieved(chunks []rag.RetrievedChunk) []rag.SourceChunk {
	sources := make([]rag.SourceChunk, 0, len(chunks))
	for i, chunk := range chunks {
		sources = append(sources, rag.SourceChunk{
			SourceNumber: i + 1,
			Path:         chunk.Path,
			SourceRoot:   chunk.SourceRoot,
			StartLine:    chunk.Chunk.StartLine,
			EndLine:      chunk.Chunk.EndLine,
			Text:         chunk.Chunk.ChunkText,
			Score:        chunk.Score,
			Chunker:      chunk.Chunk.Chunker,
			Language:     chunk.Chunk.Language,
			HeadingPath:  append([]string(nil), chunk.Chunk.HeadingPath...),
			SymbolName:   chunk.Chunk.SymbolName,
			SymbolKind:   chunk.Chunk.SymbolKind,
		})
	}
	return sources
}

func mergeBroadSearchResults(results []rag.RetrieveResult, topK int) rag.RetrieveResult {
	return mergeBroadSearchResultsForQuery("", results, topK)
}

func mergeBroadSearchResultsForQuery(originalQuery string, results []rag.RetrieveResult, topK int) rag.RetrieveResult {
	merged := rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "broad"}}
	seen := map[string]struct{}{}
	seedDocs := map[string]struct{}{}
	if len(results) != 0 {
		for _, chunk := range results[0].Chunks {
			if chunk.Chunk.DocumentID != "" {
				seedDocs[chunk.Chunk.DocumentID] = struct{}{}
			}
		}
	}
	for _, result := range results {
		if merged.Plan.Mode == "broad" && result.Plan.Mode != "" {
			merged.Plan.Mode = result.Plan.Mode
		}
		merged.Plan.Queries = append(merged.Plan.Queries, result.Plan.Queries...)
	}

	maxLen := 0
	for _, result := range results {
		if len(result.Chunks) > maxLen {
			maxLen = len(result.Chunks)
		}
	}
	for rank := 0; rank < maxLen; rank++ {
		for resultIndex, result := range results {
			if rank >= len(result.Chunks) {
				continue
			}
			chunk := result.Chunks[rank]
			if resultIndex > 0 && !keepExpansionChunk(originalQuery, seedDocs, chunk) {
				continue
			}
			key := retrievedChunkKey(chunk)
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			merged.Chunks = append(merged.Chunks, chunk)
			if topK > 0 && len(merged.Chunks) >= topK {
				return merged
			}
		}
	}
	return merged
}

func keepExpansionChunk(originalQuery string, seedDocs map[string]struct{}, chunk rag.RetrievedChunk) bool {
	if chunk.Chunk.DocumentID != "" {
		if _, ok := seedDocs[chunk.Chunk.DocumentID]; ok {
			return true
		}
	}
	haystack := strings.ToLower(chunk.Chunk.ChunkText + " " + chunk.Path)
	for _, anchor := range anchorTerms(originalQuery) {
		if strings.Contains(haystack, strings.ToLower(anchor)) {
			return true
		}
	}
	return false
}

func anchorTerms(query string) []string {
	tokens := wordTokenPattern.FindAllString(query, -1)
	anchors := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if usefulTopicTerm(token) {
			anchors = append(anchors, token)
		}
	}
	return uniqueTerms(anchors)
}

func retrievedChunkKey(chunk rag.RetrievedChunk) string {
	if chunk.Chunk.ID != "" {
		return "chunk:" + chunk.Chunk.ID
	}
	return fmt.Sprintf("fallback:%s:%d:%d:%s", chunk.Path, chunk.Chunk.StartLine, chunk.Chunk.EndLine, chunk.Chunk.ChunkText)
}

func queryPlanPayload(plan rag.QueryPlan) searchQueryPlanPayload {
	return searchQueryPlanPayload{
		Mode:    plan.Mode,
		Queries: append([]string(nil), plan.Queries...),
	}
}

func renderSearchToolContent(query string, strategy string, plan rag.QueryPlan, expansionQueries []string, suggestedQueries []string, sources []searchSourcePayload, warnings []string, newSources int, duplicateSources int, documentMatches []DocumentMatchSummary, promotedContents []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Found %d source chunks for query: %s\n", len(sources), query)
	if strategy != "" && strategy != "focused" {
		fmt.Fprintf(&b, "Strategy: %s\n", strategy)
	}
	if len(warnings) != 0 {
		b.WriteString("\nWarnings:\n")
		for _, warning := range warnings {
			fmt.Fprintf(&b, "- %s\n", warning)
		}
	}
	if len(expansionQueries) != 0 {
		b.WriteString("\nBroad search expansion queries:\n")
		for i, expansionQuery := range expansionQueries {
			fmt.Fprintf(&b, "%d. %s\n", i+1, expansionQuery)
		}
	}
	if len(suggestedQueries) != 0 {
		b.WriteString("\nSuggested follow-up queries:\n")
		for i, suggestedQuery := range suggestedQueries {
			fmt.Fprintf(&b, "%d. %s\n", i+1, suggestedQuery)
		}
	}
	if newSources == 0 && duplicateSources > 0 {
		b.WriteString("\nThis search added no new sources; consider answering instead of searching again unless another materially different query is needed.\n")
	}
	if len(plan.Queries) != 0 {
		b.WriteString("\nQuery plan:\n")
		for i, planned := range plan.Queries {
			fmt.Fprintf(&b, "%d. %s\n", i+1, planned)
		}
	}
	if len(documentMatches) != 0 {
		b.WriteString("\nDocument matches:\n")
		for _, match := range documentMatches {
			fmt.Fprintf(&b, "- %s: %d hits", match.Path, match.HitCount)
			if match.MinStartLine > 0 && match.MaxEndLine > 0 {
				fmt.Fprintf(&b, ", lines %d-%d", match.MinStartLine, match.MaxEndLine)
			}
			if match.FileLineCount > 0 {
				fmt.Fprintf(&b, ", file %d lines", match.FileLineCount)
			}
			b.WriteString("\n")
		}
	}
	for _, source := range sources {
		fmt.Fprintf(&b, "\n[source %d] %s", source.SourceNumber, source.Path)
		if source.StartLine > 0 && source.EndLine > 0 {
			fmt.Fprintf(&b, ":%d-%d", source.StartLine, source.EndLine)
		}
		b.WriteString("\nText:\n")
		b.WriteString(source.Text)
		b.WriteString("\n")
	}
	for _, content := range promotedContents {
		b.WriteString("\n")
		b.WriteString(content)
		b.WriteString("\n")
	}
	return b.String()
}

func failedSearchToolResult(status, message string, warnings []string) falken.ToolExecutionResult {
	payload := marshalSearchToolPayload(searchToolPayload{
		Success:  false,
		Status:   status,
		Error:    message,
		Warnings: warnings,
	})
	return falken.ToolExecutionResult{
		Success: false,
		Status:  status,
		Content: message,
		Payload: payload,
		Error:   message,
	}
}

func successfulSearchToolResult(content string, payload searchToolPayload) falken.ToolExecutionResult {
	return falken.ToolExecutionResult{
		Success: true,
		Status:  payload.Status,
		Content: content,
		Payload: marshalSearchToolPayload(payload),
	}
}

func marshalSearchToolPayload(payload searchToolPayload) json.RawMessage {
	if payload.Warnings == nil {
		payload.Warnings = []string{}
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return json.RawMessage(`{"success":false,"status":"internal_error","error":"encode search_index payload"}`)
	}
	return data
}

type searchToolPayload struct {
	Success            bool                        `json:"success"`
	Status             string                      `json:"status"`
	Query              string                      `json:"query,omitempty"`
	OriginalQuery      string                      `json:"original_query,omitempty"`
	QueryNormalized    bool                        `json:"query_normalized,omitempty"`
	Strategy           string                      `json:"strategy,omitempty"`
	Retrieval          string                      `json:"retrieval,omitempty"`
	TopK               int                         `json:"top_k,omitempty"`
	QueryPlan          searchQueryPlanPayload      `json:"query_plan,omitempty"`
	SeedQueryPlan      searchQueryPlanPayload      `json:"seed_query_plan,omitempty"`
	InternalPlans      []searchQueryPlanPayload    `json:"internal_query_plans,omitempty"`
	ExpansionQueries   []string                    `json:"expansion_queries,omitempty"`
	SuggestedQueries   []string                    `json:"suggested_queries,omitempty"`
	NewSources         int                         `json:"new_sources,omitempty"`
	DuplicateSources   int                         `json:"duplicate_sources,omitempty"`
	UniqueDocuments    int                         `json:"unique_documents,omitempty"`
	RetrievalCalls     int                         `json:"retrieval_calls,omitempty"`
	Sources            []searchSourcePayload       `json:"sources,omitempty"`
	DocumentMatches    []DocumentMatchSummary      `json:"document_matches,omitempty"`
	DocumentPromotions []DocumentPromotionDecision `json:"document_promotions,omitempty"`
	Warnings           []string                    `json:"warnings"`
	Error              string                      `json:"error,omitempty"`
}

type searchQueryPlanPayload struct {
	Mode    string   `json:"mode"`
	Queries []string `json:"queries"`
}

type searchSourcePayload struct {
	SourceNumber int     `json:"source_number"`
	Path         string  `json:"path"`
	StartLine    int     `json:"start_line"`
	EndLine      int     `json:"end_line"`
	Score        float32 `json:"score"`
	ChunkID      string  `json:"chunk_id"`
	DocumentID   string  `json:"document_id"`
	Text         string  `json:"text"`
}
