package falkenvector

import (
	"context"
	"errors"

	"github.com/smasonuk/falken-vector/internal/llm"
)

type observableEmbedder struct {
	next   llm.Embedder
	emit   *eventEmitter
	config ObservabilityConfig
}

func (o observableEmbedder) EmbedText(ctx context.Context, input string) (llm.Embedding, error) {
	if o.next == nil {
		err := errors.New("embedder is required")
		o.emit.emitError(EventEmbeddingFailed, err)
		return llm.Embedding{}, err
	}
	config := normalizeObservabilityConfig(o.config)
	o.emit.emit(Event{
		Type: EventEmbeddingRequest,
		Embedding: &EmbeddingEvent{
			InputLength:  len(input),
			InputPreview: observedText(input, config, config.EmitEmbeddings),
		},
	})
	embedding, err := o.next.EmbedText(ctx, input)
	if err != nil {
		o.emit.emit(Event{
			Type:  EventEmbeddingFailed,
			Error: err.Error(),
			Embedding: &EmbeddingEvent{
				InputLength: len(input),
			},
		})
		return llm.Embedding{}, err
	}
	o.emit.emit(Event{
		Type: EventEmbeddingResponse,
		Embedding: &EmbeddingEvent{
			Model:      embedding.Model,
			Dimensions: len(embedding.Vector),
		},
	})
	return embedding, nil
}
