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

	MaxSearchCalls int
	MaxTopK        int
}

type searchIndexToolState struct {
	mu    sync.Mutex
	calls int
	opts  SearchToolOptions
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
	query := strings.TrimSpace(args.Query)
	if query == "" {
		return failedSearchToolResult("invalid_arguments", "query is required", nil), nil
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

	s.mu.Lock()
	if s.calls >= maxSearchCalls {
		s.mu.Unlock()
		return failedSearchToolResult("search_call_limit", "search_index call limit exceeded", nil), nil
	}
	s.calls++
	s.mu.Unlock()

	retrieveOpts, result, failure, ok := s.retrieveOnce(ctx, retrieveOpts, query, topK, mode)
	if !ok {
		return failure, nil
	}
	expansionQueries := []string{}
	if strategy == "broad" && len(result.Chunks) != 0 {
		seedSources := sourceChunksFromRetrieved(result.Chunks)
		expansionQueries = SuggestFollowupQueries(query, seedSources, 3)
		if len(expansionQueries) != 0 {
			results := []rag.RetrieveResult{result}
			for _, expansionQuery := range expansionQueries {
				_, expansionResult, failure, ok := s.retrieveOnce(ctx, s.opts.RetrievalDefaults, expansionQuery, topK, mode)
				if !ok {
					return failure, nil
				}
				results = append(results, expansionResult)
			}
			result = mergeBroadSearchResults(results, retrieveOpts.TopK)
		}
	}

	sources := make([]searchSourcePayload, 0, len(result.Chunks))
	for _, chunk := range result.Chunks {
		source := s.opts.Registry.Register(chunk)
		sources = append(sources, searchSourcePayload{
			SourceNumber: source.SourceNumber,
			Path:         source.Path,
			StartLine:    source.StartLine,
			EndLine:      source.EndLine,
			Score:        source.Score,
			ChunkID:      chunk.Chunk.ID,
			DocumentID:   chunk.Chunk.DocumentID,
			Text:         source.Text,
		})
	}

	payload := searchToolPayload{
		Success:   true,
		Status:    "ok",
		Query:     query,
		Strategy:  strategy,
		Retrieval: string(retrieveOpts.Mode),
		TopK:      retrieveOpts.TopK,
		QueryPlan: searchQueryPlanPayload{
			Mode:    result.Plan.Mode,
			Queries: append([]string(nil), result.Plan.Queries...),
		},
		ExpansionQueries: expansionQueries,
		Sources:          sources,
		Warnings:         warnings,
	}
	return successfulSearchToolResult(renderSearchToolContent(query, strategy, result.Plan, expansionQueries, sources, warnings), payload), nil
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

func (s *searchIndexToolState) retrieveOnce(ctx context.Context, retrieveOpts rag.RetrieveOptions, query string, topK int, mode rag.RetrievalMode) (rag.RetrieveOptions, rag.RetrieveResult, falken.ToolExecutionResult, bool) {
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
	if s.opts.ConfigureQueryPlanner != nil {
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
			StartLine:    chunk.Chunk.StartLine,
			EndLine:      chunk.Chunk.EndLine,
			Text:         chunk.Chunk.ChunkText,
			Score:        chunk.Score,
		})
	}
	return sources
}

func mergeBroadSearchResults(results []rag.RetrieveResult, topK int) rag.RetrieveResult {
	merged := rag.RetrieveResult{Plan: rag.QueryPlan{Mode: "broad"}}
	seen := map[string]struct{}{}
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
		for _, result := range results {
			if rank >= len(result.Chunks) {
				continue
			}
			chunk := result.Chunks[rank]
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

func retrievedChunkKey(chunk rag.RetrievedChunk) string {
	if chunk.Chunk.ID != "" {
		return "chunk:" + chunk.Chunk.ID
	}
	return fmt.Sprintf("fallback:%s:%d:%d:%s", chunk.Path, chunk.Chunk.StartLine, chunk.Chunk.EndLine, chunk.Chunk.ChunkText)
}

func renderSearchToolContent(query string, strategy string, plan rag.QueryPlan, expansionQueries []string, sources []searchSourcePayload, warnings []string) string {
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
	if len(plan.Queries) != 0 {
		b.WriteString("\nQuery plan:\n")
		for i, planned := range plan.Queries {
			fmt.Fprintf(&b, "%d. %s\n", i+1, planned)
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
	Success          bool                   `json:"success"`
	Status           string                 `json:"status"`
	Query            string                 `json:"query,omitempty"`
	Strategy         string                 `json:"strategy,omitempty"`
	Retrieval        string                 `json:"retrieval,omitempty"`
	TopK             int                    `json:"top_k,omitempty"`
	QueryPlan        searchQueryPlanPayload `json:"query_plan,omitempty"`
	ExpansionQueries []string               `json:"expansion_queries,omitempty"`
	Sources          []searchSourcePayload  `json:"sources,omitempty"`
	Warnings         []string               `json:"warnings"`
	Error            string                 `json:"error,omitempty"`
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
