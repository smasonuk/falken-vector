package vectorstore

import "context"

type Hit struct {
	ChunkID string
	Score   float32
}

type Store interface {
	Close() error
	Insert(ctx context.Context, vector []float32, chunkID string) (int64, error)
	Search(ctx context.Context, vector []float32, limit int) ([]Hit, error)
	Commit(ctx context.Context) error
}
