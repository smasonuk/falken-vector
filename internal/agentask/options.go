package agentask

import (
	"context"
	"encoding/json"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/llm"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
)

type Options struct {
	Question string

	Paths config.Paths
	Store manifest.Store

	RetrievalDefaults rag.RetrieveOptions

	AgentLLM falken.LLM

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

	CitationPolicy rag.CitationPolicy

	MaxSearchCalls int
	MaxToolTopK    int

	// CoverageNudge controls broad-question coverage retries. Nil uses the default-on policy.
	CoverageNudge       *bool
	MinBroadSearchCalls int
	MaxCoverageRetries  int

	EnableReadSourceTool bool

	Events falken.EventSink
}

type Result struct {
	Answer           string
	Sources          []rag.SourceChunk
	CitationWarnings []string
	CitationValid    bool
	CoverageWarnings []string
	CoverageNudged   bool
	Retried          bool
	ToolCalls        []string
	Trace            AgentTrace
}

// AgentTrace records structured tool activity for an agent ask run.
type AgentTrace struct {
	ToolCalls   []ToolCallRecord
	ToolResults []ToolResultRecord
}

// ToolCallRecord captures one model-requested tool invocation.
type ToolCallRecord struct {
	Name      string
	Arguments json.RawMessage
}

// ToolResultRecord captures one tool execution result returned to the model.
type ToolResultRecord struct {
	Name    string
	Status  string
	Success bool
	Payload json.RawMessage
}
