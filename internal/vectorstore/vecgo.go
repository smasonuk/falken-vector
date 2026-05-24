package vectorstore

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"

	"github.com/hupe1980/vecgo"
)

type VecgoStore struct {
	db *vecgo.DB
}

func Open(ctx context.Context, path string, dimensions int) (Store, error) {
	if dimensions <= 0 {
		return nil, errors.New("vector dimensions must be > 0")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}

	var opts []vecgo.Option
	if !vecgoExists(path) {
		opts = append(opts, vecgo.Create(dimensions, vecgo.MetricCosine))
	}
	db, err := vecgo.Open(ctx, vecgo.Local(path), opts...)
	if err != nil {
		return nil, err
	}
	return &VecgoStore{db: db}, nil
}

func (s *VecgoStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *VecgoStore) Insert(ctx context.Context, vector []float32, chunkID string) (int64, error) {
	payload, err := json.Marshal(payload{ChunkID: chunkID})
	if err != nil {
		return 0, err
	}
	id, err := s.db.Insert(ctx, vector, nil, payload)
	if err != nil {
		return 0, err
	}
	return int64(id), nil
}

func (s *VecgoStore) Search(ctx context.Context, vector []float32, limit int) ([]Hit, error) {
	if limit <= 0 {
		return nil, nil
	}
	candidates, err := s.db.Search(ctx, vector, limit, vecgo.WithPayload())
	if err != nil {
		return nil, err
	}
	hits := make([]Hit, 0, len(candidates))
	for _, candidate := range candidates {
		var data payload
		if err := json.Unmarshal(candidate.Payload, &data); err != nil || data.ChunkID == "" {
			continue
		}
		hits = append(hits, Hit{ChunkID: data.ChunkID, Score: candidate.Score})
	}
	return hits, nil
}

func (s *VecgoStore) Commit(ctx context.Context) error {
	return s.db.Commit(ctx)
}

type payload struct {
	ChunkID string `json:"chunk_id"`
}

func vecgoExists(path string) bool {
	if _, err := os.Stat(filepath.Join(path, "CURRENT")); err == nil {
		return true
	}
	return false
}
