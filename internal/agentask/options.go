package agentask

import (
	"context"

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

	EnableReadSourceTool bool

	Events falken.EventSink
}

type Result struct {
	Answer           string
	Sources          []rag.SourceChunk
	CitationWarnings []string
	CitationValid    bool
	Retried          bool
	ToolCalls        []string
}
