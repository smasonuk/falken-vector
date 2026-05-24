package llm

import "context"

type Embedder interface {
	EmbedText(ctx context.Context, input string) (Embedding, error)
}

type Embedding struct {
	Model  string
	Vector []float32
}

type Client interface {
	Complete(ctx context.Context, req CompletionRequest) (CompletionResponse, error)
}

type CompletionRequest struct {
	Model       string
	System      string
	User        string
	Temperature float32
}

type CompletionResponse struct {
	Model string
	Text  string
}
