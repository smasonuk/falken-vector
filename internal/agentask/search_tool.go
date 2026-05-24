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

	s.mu.Lock()
	if s.calls >= maxSearchCalls {
		s.mu.Unlock()
		return failedSearchToolResult("search_call_limit", "search_index call limit exceeded", nil), nil
	}
	s.calls++
	s.mu.Unlock()

	retrieveOpts.Question = query
	retrieveOpts.TopK = topK
	retrieveOpts.Mode = mode
	retrieveOpts.Paths = s.opts.Paths

	if err := rag.CheckIndexForMode(retrieveOpts.Paths, retrieveOpts.Mode); err != nil {
		return failedSearchToolResult("index_unavailable", err.Error(), nil), nil
	}
	if rag.RetrievalModeUsesLexical(retrieveOpts.Mode) && s.opts.PrepareLexicalIndex != nil {
		if err := s.opts.PrepareLexicalIndex(ctx, s.opts.Store, retrieveOpts.Mode); err != nil {
			return failedSearchToolResult("prepare_lexical_index_failed", err.Error(), nil), nil
		}
	}
	if rag.RetrievalModeUsesVector(retrieveOpts.Mode) {
		if s.opts.EmbedderFactory == nil {
			return failedSearchToolResult("missing_embedder_factory", "embedder factory is required for vector retrieval", nil), nil
		}
		embedder, err := s.opts.EmbedderFactory()
		if err != nil {
			return failedSearchToolResult("configure_embedder_failed", err.Error(), nil), nil
		}
		retrieveOpts.Embedder = embedder
	}
	if s.opts.ConfigureQueryPlanner != nil {
		if err := s.opts.ConfigureQueryPlanner(&retrieveOpts); err != nil {
			return failedSearchToolResult("configure_query_planner_failed", err.Error(), nil), nil
		}
	}
	retrieveOpts, err = rag.NormalizeRetrieveOptions(retrieveOpts)
	if err != nil {
		return failedSearchToolResult("invalid_options", err.Error(), nil), nil
	}
	retrieveWithPlan := s.opts.RetrieveWithPlan
	if retrieveWithPlan == nil {
		retrieveWithPlan = rag.RetrieveWithPlan
	}
	result, err := retrieveWithPlan(ctx, s.opts.Store, retrieveOpts)
	if err != nil {
		return failedSearchToolResult("retrieval_failed", err.Error(), nil), nil
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
		Retrieval: string(retrieveOpts.Mode),
		TopK:      retrieveOpts.TopK,
		QueryPlan: searchQueryPlanPayload{
			Mode:    result.Plan.Mode,
			Queries: append([]string(nil), result.Plan.Queries...),
		},
		Sources:  sources,
		Warnings: warnings,
	}
	return successfulSearchToolResult(renderSearchToolContent(query, result.Plan, sources, warnings), payload), nil
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
	if topK <= 0 {
		topK = defaultTopK
	}
	if topK <= 0 {
		topK = 8
	}
	if maxTopK <= 0 {
		maxTopK = 20
	}
	if topK > maxTopK {
		return maxTopK, []string{fmt.Sprintf("top_k capped at %d", maxTopK)}
	}
	return topK, nil
}

func normalizeSearchMode(value string, defaultMode rag.RetrievalMode) (rag.RetrievalMode, error) {
	if strings.TrimSpace(value) == "" {
		return rag.ParseRetrievalMode(string(defaultMode))
	}
	return rag.ParseRetrievalMode(value)
}

func renderSearchToolContent(query string, plan rag.QueryPlan, sources []searchSourcePayload, warnings []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Found %d source chunks for query: %s\n", len(sources), query)
	if len(warnings) != 0 {
		b.WriteString("\nWarnings:\n")
		for _, warning := range warnings {
			fmt.Fprintf(&b, "- %s\n", warning)
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
	Success   bool                   `json:"success"`
	Status    string                 `json:"status"`
	Query     string                 `json:"query,omitempty"`
	Retrieval string                 `json:"retrieval,omitempty"`
	TopK      int                    `json:"top_k,omitempty"`
	QueryPlan searchQueryPlanPayload `json:"query_plan,omitempty"`
	Sources   []searchSourcePayload  `json:"sources,omitempty"`
	Warnings  []string               `json:"warnings"`
	Error     string                 `json:"error,omitempty"`
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
