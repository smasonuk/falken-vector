// Package falkenvector exposes the public Falken Vector SDK API.
package falkenvector

import "github.com/smasonuk/falken-vector/pkg/embeddings"

var (
	ErrBaseURLRequired = embeddings.ErrBaseURLRequired
	ErrModelRequired   = embeddings.ErrModelRequired
)

type Provider = embeddings.Provider
type HTTPClient = embeddings.HTTPClient
type Config = embeddings.Config
type EmbeddingRequest = embeddings.EmbeddingRequest
type EmbeddingResponse = embeddings.EmbeddingResponse
type Usage = embeddings.Usage
type Client = embeddings.Client

func New(config Config) (*Client, error) {
	return embeddings.New(config)
}
