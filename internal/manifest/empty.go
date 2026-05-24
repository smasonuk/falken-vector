package manifest

import (
	"context"
	"time"
)

type EmptyStore struct{}

func (EmptyStore) Close() error               { return nil }
func (EmptyStore) Init(context.Context) error { return nil }
func (EmptyStore) GetDocumentByPath(context.Context, string) (*Document, error) {
	return nil, ErrNotFound
}
func (EmptyStore) GetDocumentByID(context.Context, string) (*Document, error) {
	return nil, ErrNotFound
}
func (EmptyStore) UpsertDocument(context.Context, Document) error          { return nil }
func (EmptyStore) MarkDocumentError(context.Context, string, string) error { return nil }
func (EmptyStore) ListIndexedDocumentsUnderRoot(context.Context, string) ([]Document, error) {
	return nil, nil
}
func (EmptyStore) MarkDocumentDeleted(context.Context, string, time.Time) error { return nil }
func (EmptyStore) MarkDocumentsDeleted(context.Context, []string, time.Time) error {
	return nil
}
func (EmptyStore) MarkChunksInactive(context.Context, string) error                { return nil }
func (EmptyStore) InsertChunk(context.Context, Chunk) error                        { return nil }
func (EmptyStore) GetActiveChunksByIDs(context.Context, []string) ([]Chunk, error) { return nil, nil }
func (EmptyStore) Stats(context.Context) (Stats, error)                            { return Stats{}, nil }
func (EmptyStore) ListActiveIndexedChunks(context.Context, int, int) ([]Chunk, error) {
	return nil, nil
}
func (EmptyStore) UpdateChunkVectorRefs(context.Context, []ChunkVectorUpdate) error { return nil }
func (EmptyStore) SearchLexicalChunks(context.Context, string, int) ([]LexicalHit, error) {
	return nil, nil
}
func (EmptyStore) RebuildLexicalIndex(context.Context) error { return nil }
func (EmptyStore) EnsureLexicalIndex(context.Context) (bool, error) {
	return false, nil
}
