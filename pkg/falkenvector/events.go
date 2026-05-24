package falkenvector

import (
	"encoding/json"
	"time"
)

// EventSink receives SDK events. Nil sinks are treated as no-ops.
type EventSink func(Event)

// EventType identifies a public SDK event kind.
type EventType string

const (
	EventRunStarted   EventType = "run.started"
	EventRunCompleted EventType = "run.completed"
	EventRunFailed    EventType = "run.failed"

	EventRetrievalStarted   EventType = "retrieval.started"
	EventRetrievalCompleted EventType = "retrieval.completed"
	EventRetrievalFailed    EventType = "retrieval.failed"

	EventEmbeddingRequest  EventType = "embedding.request"
	EventEmbeddingResponse EventType = "embedding.response"
	EventEmbeddingFailed   EventType = "embedding.failed"

	EventLLMRequest  EventType = "llm.request"
	EventLLMResponse EventType = "llm.response"
	EventLLMDelta    EventType = "llm.delta"
	EventLLMFailed   EventType = "llm.failed"

	EventToolCall   EventType = "tool.call"
	EventToolResult EventType = "tool.result"

	EventWarning EventType = "warning"
)

// Event is the UI-facing SDK event shape.
type Event struct {
	Type EventType `json:"type"`
	At   time.Time `json:"at"`

	RunID string `json:"run_id,omitempty"`
	Seq   uint64 `json:"seq,omitempty"`

	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`

	Retrieval  *RetrievalEvent  `json:"retrieval,omitempty"`
	Embedding  *EmbeddingEvent  `json:"embedding,omitempty"`
	LLM        *LLMEvent        `json:"llm,omitempty"`
	ToolCall   *ToolCallEvent   `json:"tool_call,omitempty"`
	ToolResult *ToolResultEvent `json:"tool_result,omitempty"`
	Answer     *AnswerEvent     `json:"answer,omitempty"`
}

// ObservabilityConfig controls how much model and retrieval content is emitted.
//
// By default raw prompts, responses, embeddings, and retrieved text are omitted.
// RawPayloads defaults to false. MaxMessageBytes defaults to a safe 8 KiB limit.
type ObservabilityConfig struct {
	EmitLLMRequests   bool
	EmitLLMResponses  bool
	EmitEmbeddings    bool
	EmitRetrievalText bool

	// RawPayloads allows raw prompt, response, embedding input, and tool payload
	// previews to be emitted up to MaxMessageBytes. The default is false.
	RawPayloads bool

	// MaxMessageBytes limits emitted text payloads. The default is 8 KiB.
	MaxMessageBytes int
}

// RetrievalEvent describes retrieval lifecycle metadata.
type RetrievalEvent struct {
	Question string `json:"question,omitempty"`
	Mode     string `json:"mode,omitempty"`
	TopK     int    `json:"top_k,omitempty"`

	CandidateK        int `json:"candidate_k,omitempty"`
	VectorCandidateK  int `json:"vector_candidate_k,omitempty"`
	LexicalCandidateK int `json:"lexical_candidate_k,omitempty"`

	QueryPlan QueryPlan `json:"query_plan,omitempty"`

	ResultCount int              `json:"result_count,omitempty"`
	Sources     []RetrievedChunk `json:"sources,omitempty"`

	Warning string `json:"warning,omitempty"`
}

// EmbeddingEvent describes an embedding request or response without exposing
// vector values by default.
type EmbeddingEvent struct {
	Model        string `json:"model,omitempty"`
	InputLength  int    `json:"input_length,omitempty"`
	InputPreview string `json:"input_preview,omitempty"`
	Dimensions   int    `json:"dimensions,omitempty"`
}

// LLMEvent describes a model request, response, delta, or failure.
type LLMEvent struct {
	Label        string          `json:"label,omitempty"`
	Model        string          `json:"model,omitempty"`
	Temperature  float32         `json:"temperature,omitempty"`
	Messages     []LLMMessage    `json:"messages,omitempty"`
	Tools        []LLMTool       `json:"tools,omitempty"`
	Text         string          `json:"text,omitempty"`
	FinishReason string          `json:"finish_reason,omitempty"`
	ToolCalls    []ToolCallEvent `json:"tool_calls,omitempty"`
}

// LLMMessage is a redacted or truncated LLM message preview.
type LLMMessage struct {
	Role          string `json:"role,omitempty"`
	Content       string `json:"content,omitempty"`
	ContentLength int    `json:"content_length,omitempty"`
}

// LLMTool describes a tool made available to an LLM.
type LLMTool struct {
	Name string `json:"name,omitempty"`
}

// ToolCallEvent describes a model-requested tool invocation.
type ToolCallEvent struct {
	CallID    string          `json:"call_id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Arguments string          `json:"arguments,omitempty"`
	Payload   json.RawMessage `json:"payload,omitempty"`
	Status    string          `json:"status,omitempty"`
	Error     string          `json:"error,omitempty"`
	Success   bool            `json:"success,omitempty"`
}

// ToolResultEvent describes a tool invocation result.
type ToolResultEvent struct {
	CallID  string          `json:"call_id,omitempty"`
	Name    string          `json:"name,omitempty"`
	Content string          `json:"content,omitempty"`
	Payload json.RawMessage `json:"payload,omitempty"`
	Error   string          `json:"error,omitempty"`
	Status  string          `json:"status,omitempty"`
	Success bool            `json:"success,omitempty"`
}

// AnswerEvent describes the final answer metadata for a run.
type AnswerEvent struct {
	Text             string            `json:"text,omitempty"`
	SourceCount      int               `json:"source_count,omitempty"`
	CitationWarnings []string          `json:"citation_warnings,omitempty"`
	CitationValid    bool              `json:"citation_valid,omitempty"`
	Retried          bool              `json:"retried,omitempty"`
	ToolCalls        []ToolCallSummary `json:"tool_calls,omitempty"`
}
