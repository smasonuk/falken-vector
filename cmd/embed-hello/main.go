package main

import (
	"context"
	"fmt"
	"os"
	"time"

	falkenvector "github.com/smasonuk/falken-vector"
	"github.com/smasonuk/falken-vector/internal/llm"
)

const helloInput = "Hello from Falken vector embeddings."

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	apiKey := os.Getenv(llm.EnvEmbeddingModelAPIKey)
	if apiKey == "" {
		return fmt.Errorf("%s environment variable is required", llm.EnvEmbeddingModelAPIKey)
	}

	client, err := falkenvector.New(falkenvector.PortkeyConfig(apiKey))
	if err != nil {
		return fmt.Errorf("configure embeddings client: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	response, err := client.Embed(ctx, falkenvector.EmbeddingRequest{
		Input: helloInput,
	})
	if err != nil {
		return fmt.Errorf("retrieve embedding: %w", err)
	}

	fmt.Printf("model: %s\n", response.Model)
	fmt.Printf("dimensions: %d\n", len(response.Embedding))
	fmt.Printf("usage: prompt_tokens=%d total_tokens=%d\n", response.Usage.PromptTokens, response.Usage.TotalTokens)
	fmt.Printf("first values: %v\n", vectorPrefix(response.Embedding, 8))
	return nil
}

func vectorPrefix(vector []float64, limit int) []float64 {
	if len(vector) <= limit {
		return vector
	}
	return vector[:limit]
}
