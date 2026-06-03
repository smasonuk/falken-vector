package falkenvector

import (
	"context"
	"net/http"
	"time"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/rag"
)

// ErrNoIndex reports that no Falken Vector index exists for the configured
// state directory.
var ErrNoIndex = rag.ErrNoIndex

// ModelConfig describes an OpenAI-compatible model endpoint.
//
// BaseURL and Model must be supplied when the model is needed. Headers are
// copied before use.
type ModelConfig struct {
	APIKey  string
	BaseURL string
	Model   string
	Headers map[string]string
}

// EngineConfig configures a public Falken Vector engine.
//
// StateDir defaults to the CLI state directory when left empty.
// EngineConfigFromEnvE populates it from FALKENGO_STATE_DIR when set.
// Retrieval uses lexical mode by default when Mode is empty. Events is optional
// and nil means no events are delivered. Observability defaults are
// conservative and do not include raw prompt, embedding input, or retrieved
// chunk text. AgentLLM is optional for callers that already have a
// Falken-compatible agent model.
type EngineConfig struct {
	StateDir string

	Embedding ModelConfig
	Chat      ModelConfig

	Retrieval RetrievalOptions
	Events    EventSink

	Observability ObservabilityConfig

	HTTPClient HTTPClient

	// AgentLLM optionally injects a Falken-compatible LLM for agent mode.
	AgentLLM FalkenLLM
}

// HTTPClient is the minimal HTTP client contract used by the SDK.
type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

// RetrievalMode selects the public engine retrieval backend. Empty mode values
// in EngineConfig and request RetrievalOptions default to lexical.
type RetrievalMode string

const (
	RetrievalVector  RetrievalMode = "vector"
	RetrievalLexical RetrievalMode = "lexical"
	RetrievalHybrid  RetrievalMode = "hybrid"
)

// RerankerMode selects an optional reranking strategy. Empty defaults to none.
type RerankerMode string

const (
	RerankerNone      RerankerMode = "none"
	RerankerHeuristic RerankerMode = "heuristic"
)

// QueryPlannerMode selects optional query expansion. Empty defaults to none.
type QueryPlannerMode string

const (
	QueryPlannerNone      QueryPlannerMode = "none"
	QueryPlannerHeuristic QueryPlannerMode = "heuristic"
	QueryPlannerLLM       QueryPlannerMode = "llm"
)

// RetrievalOptions configures query-time retrieval.
//
// TopK defaults to the internal retrieval default. Candidate counts, when zero,
// are derived from TopK by the retrieval layer. IncludeGlobs, ExcludeGlobs, and
// SourceRoots narrow the source corpus when provided. Mode defaults to lexical
// in the public engine; set it explicitly when matching CLI vector defaults.
type RetrievalOptions struct {
	Mode              RetrievalMode
	Reranker          RerankerMode
	QueryPlanner      QueryPlannerMode
	TopK              int
	CandidateK        int
	VectorCandidateK  int
	LexicalCandidateK int
	MaxSubqueries     int

	IncludeGlobs []string
	ExcludeGlobs []string
	SourceRoots  []string
}

// CitationPolicy controls answer citation validation. Empty defaults to
// CitationValidateAndRetry.
type CitationPolicy string

const (
	CitationValidateAndRetry CitationPolicy = "validate-and-retry"
	CitationValidateOnly     CitationPolicy = "validate-only"
	CitationOff              CitationPolicy = "off"
)

type AgentCoveragePolicy string

const (
	AgentCoverageDefault AgentCoveragePolicy = ""
	AgentCoverageOff     AgentCoveragePolicy = "off"
	AgentCoverageOn      AgentCoveragePolicy = "on"
)

type ReadSourcePolicy string

const (
	ReadSourcePolicyDefault ReadSourcePolicy = ""
	ReadSourcePolicyAuto    ReadSourcePolicy = "auto"
	ReadSourcePolicyOn      ReadSourcePolicy = "on"
	ReadSourcePolicyOff     ReadSourcePolicy = "off"
)

type ReadSourceOverlapPolicy string

const (
	ReadSourceOverlapDefault ReadSourceOverlapPolicy = ""
	ReadSourceOverlapSkip    ReadSourceOverlapPolicy = "skip"
	ReadSourceOverlapMerge   ReadSourceOverlapPolicy = "merge"
	ReadSourceOverlapAllow   ReadSourceOverlapPolicy = "allow"
)

// AskRequest asks the engine to answer a question.
//
// Retrieval overrides engine defaults for this call. Agent enables the tool
// calling path. CitationPolicy defaults to validate-and-retry. MaxAgentSearches,
// MaxToolTopK, and MaxAgentExpansionQueries use agent defaults when zero.
// AgentCoverage controls the default broad-question coverage nudge, which asks
// the agent to perform additional searches for exploratory questions.
// ReadSourcePolicy controls read_index_source in agent mode. Default/auto enables
// it for broad questions. ReadSourceTool is preserved for compatibility and acts
// like ReadSourcePolicyOn unless ReadSourcePolicyOff is set.
// ReadSourceOverlapPolicy controls whether overlapping read_index_source calls
// are skipped, merged into an existing expanded source, or always allowed.
// The default uses merge for broad agent questions and skip otherwise.
type AskRequest struct {
	Question string

	Retrieval RetrievalOptions

	Agent bool

	CitationPolicy CitationPolicy

	MaxAgentSearches int
	MaxToolTopK      int
	// MaxAgentExpansionQueries limits internal broad-search expansion queries.
	// Zero uses the agent default.
	MaxAgentExpansionQueries int
	// MaxAgentRetrievals limits actual retrieval calls across agent searches.
	// Zero leaves retrievals uncapped.
	MaxAgentRetrievals int

	AgentCoverage      AgentCoveragePolicy
	MinAgentSearches   int
	MaxCoverageRetries int

	ReadSourceTool          bool
	ReadSourcePolicy        ReadSourcePolicy
	ReadSourceOverlapPolicy ReadSourceOverlapPolicy
	// MaxMergedReadSourceLines caps merged read_index_source ranges.
	// Zero uses the agent default.
	MaxMergedReadSourceLines int
}

// Answer is the structured answer returned by Ask.
type Answer struct {
	Text string

	// Sources contains all sources made available to the answer path.
	Sources []Source
	// CitedSources contains the subset of Sources cited by Text.
	CitedSources []Source
	// AvailableSources mirrors Sources for callers that want explicit naming.
	AvailableSources []Source

	CitationWarnings         []string
	CitationValid            bool
	CoverageWarnings         []string
	CoverageNudged           bool
	ThinSourceWarnings       []string
	ThinSourceNudged         bool
	Retried                  bool
	SearchToolCalls          int
	RetrievalCalls           int
	ReadSourceCalls          int
	ReadSourceOverlapPolicy  string
	ReadSourceAlreadyCovered int
	ReadSourceMerges         int
	ReadSourceMergeTooLarge  int

	ToolCalls []ToolCallSummary
}

// QueryRequest asks the engine to retrieve supporting chunks without answering.
type QueryRequest struct {
	Question  string
	Retrieval RetrievalOptions
}

// QueryResult is the structured retrieval result returned by Query.
type QueryResult struct {
	Question  string
	Chunks    []RetrievedChunk
	QueryPlan QueryPlan
}

// Source identifies a source span in an indexed document.
type Source struct {
	Number    int
	Path      string
	StartLine int
	EndLine   int
}

// RetrievedChunk describes an indexed chunk returned by retrieval.
type RetrievedChunk struct {
	Source

	Score      float32
	ChunkID    string
	DocumentID string
	Text       string
}

// QueryPlan describes generated retrieval queries.
type QueryPlan struct {
	Mode    string
	Queries []string
	Warning string
}

// ToolCallSummary records a tool call observed while answering.
type ToolCallSummary struct {
	Name string
	ID   string
}

// Status summarizes the current index state.
type Status struct {
	StateDir     string
	ManifestPath string
	VectorPath   string

	Documents StatusDocumentCounts
	Chunks    StatusChunkCounts

	LastIndexedAt *time.Time
}

// StatusDocumentCounts contains document status totals.
type StatusDocumentCounts struct {
	Indexed int
	Error   int
	Deleted int
}

// StatusChunkCounts contains active and inactive chunk totals.
type StatusChunkCounts struct {
	Active   int
	Inactive int
}

// ChunkerMode selects the ingest chunking strategy. Empty defaults to
// ChunkerAuto.
type ChunkerMode string

const (
	ChunkerAuto     ChunkerMode = "auto"
	ChunkerFixed    ChunkerMode = "fixed"
	ChunkerMarkdown ChunkerMode = "markdown"
	ChunkerText     ChunkerMode = "text"
	ChunkerCode     ChunkerMode = "code"
)

// IngestRequest configures a source indexing run.
//
// Root defaults to ".". ChunkSize defaults to the CLI/internal ingest default.
// ChunkOverlap defaults to the CLI default. ChunkerMode defaults to ChunkerAuto.
type IngestRequest struct {
	Root                 string
	Extensions           []string
	ExcludeExtensions    []string
	ExcludeDirs          []string
	ChunkSize            int
	ChunkOverlap         int
	ChunkerMode          ChunkerMode
	DryRun               bool
	SyncSource           bool
	EmbeddingConcurrency int
}

// IngestResult summarizes an ingest run.
type IngestResult struct {
	Directory      string
	Scanned        int
	NewFiles       int
	ChangedFiles   int
	UnchangedFiles int
	DeletedFiles   int
	FailedFiles    int
	ChunksEmbedded int
	Warnings       []string
}

// CompactRequest configures vector database compaction.
type CompactRequest struct {
	DryRun               bool
	BatchSize            int
	KeepBackup           bool
	EmbeddingConcurrency int
}

// CompactResult summarizes a compaction run.
type CompactResult struct {
	ActiveChunks     int
	InactiveChunks   int
	ReembeddedChunks int
	UpdatedChunks    int
	EmbeddingModel   string
	BackupPath       string
	PendingRuns      int
	Warnings         []string
}

// FalkenLLM is the Falken Core LLM contract accepted by agent mode.
type FalkenLLM interface {
	Complete(context.Context, falken.CompletionRequest) (falken.CompletionResponse, error)
}
