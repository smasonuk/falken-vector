package falkenvector

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/smasonuk/falken-core/pkg/falken"
	falkenlangchain "github.com/smasonuk/falken-extra/llm/langchaingo"
	"github.com/smasonuk/falken-vector/internal/agentask"
	"github.com/smasonuk/falken-vector/internal/compact"
	internalconfig "github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/smasonuk/falken-vector/pkg/embeddings"
	"github.com/tmc/langchaingo/llms/openai"
)

// Engine is the public Falken Vector SDK facade.
type Engine struct {
	config EngineConfig
	paths  internalconfig.Paths

	mu     sync.Mutex
	closed bool
}

// NewEngine validates config and creates an SDK engine without opening network
// connections or writing output.
func NewEngine(config EngineConfig) (*Engine, error) {
	config.Observability = normalizeObservabilityConfig(config.Observability)
	config.Embedding.Headers = copyHeaders(config.Embedding.Headers)
	config.Chat.Headers = copyHeaders(config.Chat.Headers)
	if config.HTTPClient == nil {
		config.HTTPClient = http.DefaultClient
	}
	if _, err := toRAGRetrievalOptions(config.Retrieval); err != nil {
		return nil, err
	}
	paths, err := internalconfig.ResolvePaths(config.StateDir)
	if err != nil {
		return nil, fmt.Errorf("resolve state paths: %w", err)
	}
	return &Engine{config: config, paths: paths}, nil
}

// EngineConfigFromEnv reads FALKENGO_* model configuration from getenv.
func EngineConfigFromEnv(getenv func(string) string) EngineConfig {
	if getenv == nil {
		getenv = func(string) string { return "" }
	}
	chatAPIKey := strings.TrimSpace(getenv(llm.EnvLLMAPIKey))
	if chatAPIKey == "" {
		chatAPIKey = strings.TrimSpace(getenv(llm.EnvEmbeddingModelAPIKey))
	}
	return EngineConfig{
		StateDir: strings.TrimSpace(getenv("FALKENGO_STATE_DIR")),
		Embedding: ModelConfig{
			APIKey:  strings.TrimSpace(getenv(llm.EnvEmbeddingModelAPIKey)),
			BaseURL: strings.TrimSpace(getenv(llm.EnvEmbeddingModelURL)),
			Model:   strings.TrimSpace(getenv(llm.EnvEmbeddingModel)),
		},
		Chat: ModelConfig{
			APIKey:  chatAPIKey,
			BaseURL: strings.TrimSpace(getenv(llm.EnvLLMBaseURL)),
			Model:   strings.TrimSpace(getenv(llm.EnvLLMModel)),
		},
	}
}

// Close releases engine resources. It is safe to call multiple times.
func (e *Engine) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.closed = true
	return nil
}

// Query retrieves chunks and a query plan without printing.
func (e *Engine) Query(ctx context.Context, request QueryRequest) (QueryResult, error) {
	result, _, err := e.query(ctx, request, newEventEmitter(e.eventSink()))
	return result, err
}

// Ask answers a question without printing. Agent mode uses Falken Core tools;
// non-agent mode retrieves chunks first and sends them to the chat model.
func (e *Engine) Ask(ctx context.Context, request AskRequest) (Answer, error) {
	if request.Agent {
		return e.askAgent(ctx, request)
	}
	return e.askRAG(ctx, request)
}

// Status returns index paths and counts without printing.
func (e *Engine) Status(ctx context.Context) (Status, error) {
	if err := e.checkReady(); err != nil {
		return Status{}, err
	}
	if _, err := os.Stat(e.paths.ManifestPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Status{}, rag.ErrNoIndex
		}
		return Status{}, err
	}
	store, err := manifest.Open(e.paths.ManifestPath)
	if err != nil {
		return Status{}, fmt.Errorf("open manifest database: %w", err)
	}
	defer store.Close()
	stats, err := store.Stats(ctx)
	if err != nil {
		return Status{}, fmt.Errorf("read manifest stats: %w", err)
	}
	return Status{
		StateDir:     e.paths.StateDir,
		ManifestPath: e.paths.ManifestPath,
		VectorPath:   e.paths.VecgoPath,
		Documents: StatusDocumentCounts{
			Indexed: stats.IndexedDocuments,
			Error:   stats.ErrorDocuments,
			Deleted: stats.DeletedDocuments,
		},
		Chunks: StatusChunkCounts{
			Active:   stats.ActiveChunks,
			Inactive: stats.InactiveChunks,
		},
		LastIndexedAt: stats.LastIndexedAt,
	}, nil
}

// Compact rebuilds the vector database from active manifest chunks without
// printing. Embedding progress is emitted through Events when configured.
func (e *Engine) Compact(ctx context.Context, request CompactRequest) (CompactResult, error) {
	if err := e.checkReady(); err != nil {
		return CompactResult{}, err
	}
	emitter := newEventEmitter(e.eventSink())
	emitter.emit(Event{Type: EventRunStarted, Message: "compact"})
	if _, err := os.Stat(e.paths.ManifestPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			emitter.emitError(EventRunFailed, rag.ErrNoIndex)
			return CompactResult{}, rag.ErrNoIndex
		}
		err = fmt.Errorf("inspect manifest database: %w", err)
		emitter.emitError(EventRunFailed, err)
		return CompactResult{}, err
	}
	var lock *internalconfig.WriteLock
	var err error
	if !request.DryRun {
		lock, err = internalconfig.AcquireWriteLock(e.paths)
		if err != nil {
			err = fmt.Errorf("acquire write lock: %w", err)
			emitter.emitError(EventRunFailed, err)
			return CompactResult{}, err
		}
		defer lock.Release()
	}
	store, err := manifest.Open(e.paths.ManifestPath)
	if err != nil {
		err = fmt.Errorf("open manifest database: %w", err)
		emitter.emitError(EventRunFailed, err)
		return CompactResult{}, err
	}
	defer store.Close()
	if !request.DryRun {
		if err := store.Init(ctx); err != nil {
			err = fmt.Errorf("initialize manifest database: %w", err)
			emitter.emitError(EventRunFailed, err)
			return CompactResult{}, err
		}
	}
	var embedder llm.Embedder
	if !request.DryRun {
		embedder, err = e.newEmbedder()
		if err != nil {
			err = fmt.Errorf("configure embedder: %w", err)
			emitter.emitError(EventRunFailed, err)
			return CompactResult{}, err
		}
		embedder = observableEmbedder{next: embedder, emit: emitter, config: e.config.Observability}
	}
	summary, err := compact.Run(ctx, store, compact.Options{
		Paths:      e.paths,
		Embedder:   embedder,
		DryRun:     request.DryRun,
		KeepBackup: request.KeepBackup,
		BatchSize:  request.BatchSize,
		Out:        io.Discard,
	})
	result := compactResult(summary, request.DryRun)
	for _, warning := range result.Warnings {
		emitter.emit(Event{Type: EventWarning, Message: warning})
	}
	if err != nil {
		emitter.emitError(EventRunFailed, err)
		return result, err
	}
	emitter.emit(Event{
		Type:    EventRunCompleted,
		Message: fmt.Sprintf("compacted %d active chunks", result.ActiveChunks),
	})
	return result, nil
}

func (e *Engine) askRAG(ctx context.Context, request AskRequest) (Answer, error) {
	emitter := newEventEmitter(e.eventSink())
	emitter.emit(Event{Type: EventRunStarted, Message: "ask"})
	result, chunks, err := e.query(ctx, QueryRequest{Question: request.Question, Retrieval: request.Retrieval}, emitter)
	if err != nil {
		emitter.emitError(EventRunFailed, err)
		return Answer{}, err
	}
	if len(chunks) == 0 {
		answer := Answer{CitationValid: true}
		emitter.emit(Event{
			Type:   EventRunCompleted,
			Answer: answerEvent(answer, e.config.Observability),
		})
		return answer, nil
	}
	client, err := e.newChatClient()
	if err != nil {
		err = fmt.Errorf("configure LLM client: %w", err)
		emitter.emitError(EventRunFailed, err)
		return Answer{}, err
	}
	askResult, err := rag.Ask(ctx, rag.AskOptions{
		Question:       request.Question,
		Model:          e.config.Chat.Model,
		Chunks:         chunks,
		LLM:            observableInternalLLM{next: client, emit: emitter, config: e.config.Observability, label: "answer"},
		CitationPolicy: toRAGCitationPolicy(request.CitationPolicy),
	})
	if err != nil {
		emitter.emitError(EventRunFailed, err)
		return Answer{}, err
	}
	answer := answerFromRAG(askResult)
	emitter.emit(Event{
		Type:   EventRunCompleted,
		Answer: answerEvent(answer, e.config.Observability),
	})
	_ = result
	return answer, nil
}

func (e *Engine) askAgent(ctx context.Context, request AskRequest) (Answer, error) {
	if strings.TrimSpace(request.Question) == "" {
		return Answer{}, errors.New("question is required")
	}
	if err := e.checkReady(); err != nil {
		return Answer{}, err
	}
	emitter := newEventEmitter(e.eventSink())
	emitter.emit(Event{Type: EventRunStarted, Message: "ask.agent"})
	store, err := manifest.Open(e.paths.ManifestPath)
	if err != nil {
		err = fmt.Errorf("open manifest database: %w", err)
		emitter.emitError(EventRunFailed, err)
		return Answer{}, err
	}
	defer store.Close()

	retrieval, err := e.ragRetrievalOptions(request.Retrieval)
	if err != nil {
		emitter.emitError(EventRunFailed, err)
		return Answer{}, err
	}
	retrieval.Question = request.Question
	retrieval.Paths = e.paths
	if err := checkAgentManifest(e.paths.ManifestPath); err != nil {
		emitter.emitError(EventRunFailed, err)
		return Answer{}, err
	}
	agentLLM, err := e.agentLLM()
	if err != nil {
		err = fmt.Errorf("configure agent LLM: %w", err)
		emitter.emitError(EventRunFailed, err)
		return Answer{}, err
	}
	result, err := agentask.Run(ctx, agentask.Options{
		Question:              request.Question,
		Paths:                 e.paths,
		Store:                 store,
		RetrievalDefaults:     retrieval,
		AgentLLM:              observableFalkenLLM{next: agentLLM, emit: emitter, config: e.config.Observability},
		EmbedderFactory:       e.observableEmbedderFactory(emitter),
		RetrieveWithPlan:      e.observableRetrieveWithPlan(emitter),
		PrepareLexicalIndex:   prepareEngineLexicalIndex,
		ConfigureQueryPlanner: e.configureQueryPlanner(emitter),
		CitationPolicy:        toRAGCitationPolicy(request.CitationPolicy),
		MaxSearchCalls:        request.MaxAgentSearches,
		MaxToolTopK:           request.MaxToolTopK,
		EnableReadSourceTool:  request.ReadSourceTool,
		Events: func(event falken.Event) {
			converted := convertFalkenEvent(event)
			if converted.Type == EventRunCompleted || converted.Type == EventRunFailed {
				return
			}
			emitter.emit(converted)
		},
	})
	if err != nil {
		err = fmt.Errorf("ask agent: %w", err)
		emitter.emitError(EventRunFailed, err)
		return Answer{}, err
	}
	answer := answerFromAgent(result)
	emitter.emit(Event{
		Type:   EventRunCompleted,
		Answer: answerEvent(answer, e.config.Observability),
	})
	return answer, nil
}

func (e *Engine) query(ctx context.Context, request QueryRequest, emitter *eventEmitter) (QueryResult, []rag.RetrievedChunk, error) {
	if strings.TrimSpace(request.Question) == "" {
		return QueryResult{}, nil, errors.New("question is required")
	}
	if err := e.checkReady(); err != nil {
		return QueryResult{}, nil, err
	}
	opts, err := e.ragRetrievalOptions(request.Retrieval)
	if err != nil {
		return QueryResult{}, nil, err
	}
	opts.Question = request.Question
	opts.Paths = e.paths
	opts, err = rag.NormalizeRetrieveOptions(opts)
	if err != nil {
		return QueryResult{}, nil, err
	}
	start := retrievalEvent(opts, nil, e.config.Observability)
	emitter.emit(Event{Type: EventRetrievalStarted, Retrieval: start})
	if err := rag.CheckIndexForMode(e.paths, opts.Mode); err != nil {
		emitter.emit(Event{Type: EventRetrievalFailed, Error: err.Error(), Retrieval: start})
		return QueryResult{}, nil, err
	}
	store, err := manifest.Open(e.paths.ManifestPath)
	if err != nil {
		err = fmt.Errorf("open manifest database: %w", err)
		emitter.emit(Event{Type: EventRetrievalFailed, Error: err.Error(), Retrieval: start})
		return QueryResult{}, nil, err
	}
	defer store.Close()
	if err := prepareEngineLexicalIndex(ctx, store, opts.Mode); err != nil {
		emitter.emit(Event{Type: EventRetrievalFailed, Error: err.Error(), Retrieval: start})
		return QueryResult{}, nil, err
	}
	if err := e.configureQueryPlanner(emitter)(&opts); err != nil {
		emitter.emit(Event{Type: EventRetrievalFailed, Error: err.Error(), Retrieval: start})
		return QueryResult{}, nil, err
	}
	if rag.RetrievalModeUsesVector(opts.Mode) {
		embedder, err := e.newEmbedder()
		if err != nil {
			err = fmt.Errorf("configure embedder: %w", err)
			emitter.emit(Event{Type: EventRetrievalFailed, Error: err.Error(), Retrieval: start})
			return QueryResult{}, nil, err
		}
		opts.Embedder = observableEmbedder{next: embedder, emit: emitter, config: e.config.Observability}
	}
	retrievalResult, err := rag.RetrieveWithPlan(ctx, store, opts)
	if err != nil {
		emitter.emit(Event{Type: EventRetrievalFailed, Error: err.Error(), Retrieval: start})
		return QueryResult{}, nil, err
	}
	result := QueryResult{
		Question:  request.Question,
		Chunks:    publicRetrievedChunks(retrievalResult.Chunks, e.config.Observability, true),
		QueryPlan: publicQueryPlan(retrievalResult.Plan),
	}
	completed := retrievalEvent(opts, &retrievalResult, e.config.Observability)
	emitter.emit(Event{Type: EventRetrievalCompleted, Retrieval: completed})
	return result, retrievalResult.Chunks, nil
}

func (e *Engine) checkReady() error {
	if e == nil {
		return errors.New("engine is nil")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("engine is closed")
	}
	return nil
}

func (e *Engine) eventSink() EventSink {
	if e == nil {
		return nil
	}
	return e.config.Events
}

func (e *Engine) ragRetrievalOptions(override RetrievalOptions) (rag.RetrieveOptions, error) {
	return toRAGRetrievalOptions(mergeRetrievalOptions(e.config.Retrieval, override))
}

func toRAGRetrievalOptions(options RetrievalOptions) (rag.RetrieveOptions, error) {
	mode := string(options.Mode)
	if strings.TrimSpace(mode) == "" {
		mode = string(RetrievalLexical)
	}
	parsedMode, err := rag.ParseRetrievalMode(mode)
	if err != nil {
		return rag.RetrieveOptions{}, err
	}
	reranker := string(options.Reranker)
	if strings.TrimSpace(reranker) == "" {
		reranker = string(RerankerNone)
	}
	parsedReranker, err := rag.ParseRerankerMode(reranker)
	if err != nil {
		return rag.RetrieveOptions{}, err
	}
	planner := string(options.QueryPlanner)
	if strings.TrimSpace(planner) == "" {
		planner = string(QueryPlannerNone)
	}
	parsedPlanner, err := rag.ParseQueryPlannerMode(planner)
	if err != nil {
		return rag.RetrieveOptions{}, err
	}
	sourceFilter, err := parseSourceFilter(options)
	if err != nil {
		return rag.RetrieveOptions{}, err
	}
	return rag.RetrieveOptions{
		Mode:              parsedMode,
		RerankerMode:      parsedReranker,
		QueryPlannerMode:  parsedPlanner,
		TopK:              options.TopK,
		CandidateK:        options.CandidateK,
		VectorCandidateK:  options.VectorCandidateK,
		LexicalCandidateK: options.LexicalCandidateK,
		MaxSubqueries:     options.MaxSubqueries,
		SourceFilter:      sourceFilter,
	}, nil
}

func mergeRetrievalOptions(base RetrievalOptions, override RetrievalOptions) RetrievalOptions {
	out := base
	if override.Mode != "" {
		out.Mode = override.Mode
	}
	if override.Reranker != "" {
		out.Reranker = override.Reranker
	}
	if override.QueryPlanner != "" {
		out.QueryPlanner = override.QueryPlanner
	}
	if override.TopK != 0 {
		out.TopK = override.TopK
	}
	if override.CandidateK != 0 {
		out.CandidateK = override.CandidateK
	}
	if override.VectorCandidateK != 0 {
		out.VectorCandidateK = override.VectorCandidateK
	}
	if override.LexicalCandidateK != 0 {
		out.LexicalCandidateK = override.LexicalCandidateK
	}
	if override.MaxSubqueries != 0 {
		out.MaxSubqueries = override.MaxSubqueries
	}
	if override.IncludeGlobs != nil {
		out.IncludeGlobs = append([]string(nil), override.IncludeGlobs...)
	}
	if override.ExcludeGlobs != nil {
		out.ExcludeGlobs = append([]string(nil), override.ExcludeGlobs...)
	}
	if override.SourceRoots != nil {
		out.SourceRoots = append([]string(nil), override.SourceRoots...)
	}
	return out
}

func parseSourceFilter(options RetrievalOptions) (rag.SourceFilter, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return rag.SourceFilter{}, fmt.Errorf("get current directory: %w", err)
	}
	return rag.ParseSourceFilter(options.IncludeGlobs, options.ExcludeGlobs, options.SourceRoots, cwd)
}

func (e *Engine) configureQueryPlanner(emitter *eventEmitter) func(*rag.RetrieveOptions) error {
	return func(opts *rag.RetrieveOptions) error {
		if opts.QueryPlannerMode != rag.QueryPlannerModeLLM {
			return nil
		}
		client, err := e.newChatClient()
		if err != nil {
			return fmt.Errorf("configure query planner LLM: %w", err)
		}
		opts.QueryPlanner = rag.LLMQueryPlanner{
			LLM:   observableInternalLLM{next: client, emit: emitter, config: e.config.Observability, label: "query-planner"},
			Model: e.config.Chat.Model,
		}
		return nil
	}
}

func (e *Engine) observableEmbedderFactory(emitter *eventEmitter) func() (llm.Embedder, error) {
	return func() (llm.Embedder, error) {
		embedder, err := e.newEmbedder()
		if err != nil {
			return nil, err
		}
		return observableEmbedder{next: embedder, emit: emitter, config: e.config.Observability}, nil
	}
}

func (e *Engine) observableRetrieveWithPlan(emitter *eventEmitter) func(context.Context, manifest.Store, rag.RetrieveOptions) (rag.RetrieveResult, error) {
	return func(ctx context.Context, store manifest.Store, opts rag.RetrieveOptions) (rag.RetrieveResult, error) {
		normalized, err := rag.NormalizeRetrieveOptions(opts)
		if err != nil {
			return rag.RetrieveResult{}, err
		}
		start := retrievalEvent(normalized, nil, e.config.Observability)
		emitter.emit(Event{Type: EventRetrievalStarted, Retrieval: start})
		result, err := rag.RetrieveWithPlan(ctx, store, normalized)
		if err != nil {
			emitter.emit(Event{Type: EventRetrievalFailed, Error: err.Error(), Retrieval: start})
			return rag.RetrieveResult{}, err
		}
		emitter.emit(Event{Type: EventRetrievalCompleted, Retrieval: retrievalEvent(normalized, &result, e.config.Observability)})
		return result, nil
	}
}

func prepareEngineLexicalIndex(ctx context.Context, store manifest.Store, mode rag.RetrievalMode) error {
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

func (e *Engine) newEmbedder() (llm.Embedder, error) {
	config := e.embeddingConfig()
	client, err := embeddings.New(embeddings.Config{
		BaseURL:    config.BaseURL,
		APIKey:     config.APIKey,
		Model:      config.Model,
		Headers:    config.Headers,
		HTTPClient: e.config.HTTPClient,
	})
	if err != nil {
		return nil, err
	}
	return llm.NewEmbedder(client), nil
}

func (e *Engine) newChatClient() (llm.Client, error) {
	config := e.chatConfig()
	return llm.NewChatClient(llm.ChatConfig{
		BaseURL:    config.BaseURL,
		APIKey:     config.APIKey,
		Model:      config.Model,
		Headers:    config.Headers,
		HTTPClient: e.config.HTTPClient,
	})
}

func (e *Engine) agentLLM() (falken.LLM, error) {
	if e.config.AgentLLM != nil {
		return e.config.AgentLLM, nil
	}
	config := e.chatConfig()
	if strings.TrimSpace(config.APIKey) == "" {
		return nil, llm.ErrMissingAPIKey
	}
	options := []openai.Option{
		openai.WithToken(config.APIKey),
		openai.WithModel(config.Model),
		openai.WithBaseURL(config.BaseURL),
	}
	if e.config.HTTPClient != nil || len(config.Headers) != 0 {
		options = append(options, openai.WithHTTPClient(falkenlangchain.NewHeaderHTTPClient(e.config.HTTPClient, config.Headers)))
	}
	model, err := openai.New(options...)
	if err != nil {
		return nil, fmt.Errorf("configure OpenAI-compatible agent LLM: %w", err)
	}
	return falkenlangchain.New(model), nil
}

func (e *Engine) embeddingConfig() ModelConfig {
	config := e.config.Embedding
	config.BaseURL = strings.TrimSpace(config.BaseURL)
	if config.BaseURL == "" {
		config.BaseURL = embeddings.DefaultPortkeyBaseURL
	}
	config.Model = strings.TrimSpace(config.Model)
	if config.Model == "" {
		config.Model = embeddings.DefaultEmbeddingModel
	}
	config.Headers = defaultedHeaders(config.BaseURL, config.Headers)
	return config
}

func (e *Engine) chatConfig() ModelConfig {
	config := e.config.Chat
	if strings.TrimSpace(config.APIKey) == "" {
		config.APIKey = e.config.Embedding.APIKey
	}
	config.BaseURL = strings.TrimSpace(config.BaseURL)
	if config.BaseURL == "" {
		config.BaseURL = embeddings.DefaultPortkeyBaseURL
	}
	config.Model = strings.TrimSpace(config.Model)
	if config.Model == "" {
		config.Model = llm.DefaultChatModel
	}
	config.Headers = defaultedHeaders(config.BaseURL, config.Headers)
	return config
}

func defaultedHeaders(baseURL string, headers map[string]string) map[string]string {
	out := copyHeaders(headers)
	if isDefaultPortkeyBaseURL(baseURL) {
		if _, ok := out["X-Portkey-Provider"]; !ok {
			out["X-Portkey-Provider"] = embeddings.DefaultPortkeyProvider
		}
	}
	return out
}

func isDefaultPortkeyBaseURL(baseURL string) bool {
	return strings.TrimRight(strings.TrimSpace(baseURL), "/") == strings.TrimRight(embeddings.DefaultPortkeyBaseURL, "/")
}

func copyHeaders(headers map[string]string) map[string]string {
	out := make(map[string]string, len(headers))
	for name, value := range headers {
		name = strings.TrimSpace(name)
		value = strings.TrimSpace(value)
		if name != "" && value != "" {
			out[name] = value
		}
	}
	return out
}

func toRAGCitationPolicy(policy CitationPolicy) rag.CitationPolicy {
	switch policy {
	case CitationValidateOnly:
		return rag.CitationPolicyValidateOnly
	case CitationOff:
		return rag.CitationPolicyOff
	case "", CitationValidateAndRetry:
		return rag.CitationPolicyValidateAndRetry
	default:
		return rag.CitationPolicyValidateAndRetry
	}
}

func publicRetrievedChunks(chunks []rag.RetrievedChunk, config ObservabilityConfig, rawText bool) []RetrievedChunk {
	out := make([]RetrievedChunk, 0, len(chunks))
	for i, chunk := range chunks {
		text := chunk.Chunk.ChunkText
		if !rawText {
			text = observedText(text, config, config.EmitRetrievalText)
		}
		out = append(out, RetrievedChunk{
			Source: Source{
				Number:    i + 1,
				Path:      chunk.Path,
				StartLine: chunk.Chunk.StartLine,
				EndLine:   chunk.Chunk.EndLine,
			},
			Score:      chunk.Score,
			ChunkID:    chunk.Chunk.ID,
			DocumentID: chunk.Chunk.DocumentID,
			Text:       text,
		})
	}
	return out
}

func publicQueryPlan(plan rag.QueryPlan) QueryPlan {
	return QueryPlan{
		Mode:    plan.Mode,
		Queries: append([]string(nil), plan.Queries...),
		Warning: plan.Warning,
	}
}

func retrievalEvent(opts rag.RetrieveOptions, result *rag.RetrieveResult, config ObservabilityConfig) *RetrievalEvent {
	event := &RetrievalEvent{
		Question:          opts.Question,
		Mode:              string(opts.Mode),
		TopK:              opts.TopK,
		CandidateK:        opts.CandidateK,
		VectorCandidateK:  opts.VectorCandidateK,
		LexicalCandidateK: opts.LexicalCandidateK,
	}
	if result == nil {
		return event
	}
	event.QueryPlan = publicQueryPlan(result.Plan)
	event.ResultCount = len(result.Chunks)
	event.Sources = publicRetrievedChunks(result.Chunks, config, false)
	event.Warning = result.Plan.Warning
	return event
}

func answerFromRAG(result rag.AskResult) Answer {
	return Answer{
		Text:             result.Answer,
		Sources:          publicSources(result.Sources),
		CitationWarnings: append([]string(nil), result.CitationWarnings...),
		CitationValid:    result.CitationValid,
		Retried:          result.Retried,
	}
}

func answerFromAgent(result agentask.Result) Answer {
	calls := make([]ToolCallSummary, 0, len(result.ToolCalls))
	for _, name := range result.ToolCalls {
		calls = append(calls, ToolCallSummary{Name: name})
	}
	return Answer{
		Text:             result.Answer,
		Sources:          publicSources(result.Sources),
		CitationWarnings: append([]string(nil), result.CitationWarnings...),
		CitationValid:    result.CitationValid,
		Retried:          result.Retried,
		ToolCalls:        calls,
	}
}

func publicSources(sources []rag.SourceChunk) []Source {
	out := make([]Source, 0, len(sources))
	for _, source := range sources {
		out = append(out, Source{
			Number:    source.SourceNumber,
			Path:      source.Path,
			StartLine: source.StartLine,
			EndLine:   source.EndLine,
		})
	}
	return out
}

func answerEvent(answer Answer, config ObservabilityConfig) *AnswerEvent {
	return &AnswerEvent{
		Text:             observedText(answer.Text, config, config.EmitLLMResponses),
		SourceCount:      len(answer.Sources),
		CitationWarnings: append([]string(nil), answer.CitationWarnings...),
		CitationValid:    answer.CitationValid,
		Retried:          answer.Retried,
		ToolCalls:        append([]ToolCallSummary(nil), answer.ToolCalls...),
	}
}

func compactResult(summary compact.Summary, dryRun bool) CompactResult {
	result := CompactResult{
		ActiveChunks:     summary.ActiveChunks,
		InactiveChunks:   summary.InactiveChunks,
		ReembeddedChunks: summary.ReembeddedChunks,
		UpdatedChunks:    summary.UpdatedChunks,
		EmbeddingModel:   summary.EmbeddingModel,
		BackupPath:       summary.BackupPath,
		PendingRuns:      summary.PendingRuns,
	}
	if dryRun && summary.PendingRuns > 0 {
		result.Warnings = append(result.Warnings, "pending ingest run detected; real compact is blocked until repair is run")
	}
	return result
}

func checkAgentManifest(manifestPath string) error {
	if _, err := os.Stat(manifestPath); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return rag.ErrNoIndex
		}
		return err
	}
	return nil
}
