package ingest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"

	"github.com/smasonuk/falken-vector/internal/manifest"
)

type FileDecision string

const (
	FileDecisionNew       FileDecision = "new"
	FileDecisionChanged   FileDecision = "changed"
	FileDecisionUnchanged FileDecision = "unchanged"
)

type ChunkConfig struct {
	Chunker          string
	ChunkSize        int
	ChunkOverlap     int
	IndexTextVersion int
}

func HashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func HashText(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

func DecideFile(ctx context.Context, store manifest.Store, candidate CandidateFile) (FileDecision, string, error) {
	return DecideFileWithConfig(ctx, store, candidate, ChunkConfig{})
}

func DecideFileWithConfig(ctx context.Context, store manifest.Store, candidate CandidateFile, config ChunkConfig) (FileDecision, string, error) {
	hash, err := HashFile(candidate.Path)
	if err != nil {
		return "", "", err
	}
	doc, err := store.GetDocumentByPath(ctx, candidate.Path)
	if errors.Is(err, manifest.ErrNotFound) {
		return FileDecisionNew, hash, nil
	}
	if err != nil {
		return "", "", err
	}
	if doc.ContentHash == hash && doc.Status == manifest.DocumentStatusIndexed && documentChunkConfigMatches(doc, config) {
		return FileDecisionUnchanged, hash, nil
	}
	return FileDecisionChanged, hash, nil
}

func documentChunkConfigMatches(doc *manifest.Document, config ChunkConfig) bool {
	if config.Chunker == "" && config.ChunkSize == 0 && config.ChunkOverlap == 0 && config.IndexTextVersion == 0 {
		return true
	}
	return doc.Chunker == config.Chunker &&
		doc.ChunkSize == config.ChunkSize &&
		doc.ChunkOverlap == config.ChunkOverlap &&
		doc.IndexTextVersion == config.IndexTextVersion
}
