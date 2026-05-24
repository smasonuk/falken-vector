package manifest

import "time"

const (
	DocumentStatusIndexed = "indexed"
	DocumentStatusError   = "error"
	DocumentStatusDeleted = "deleted"
	DocumentStatusPending = "pending"
	DocumentStatusSkipped = "skipped"
)

type Document struct {
	ID               string
	Path             string
	ContentHash      string
	SizeBytes        int64
	ModifiedAt       time.Time
	IndexedAt        *time.Time
	DeletedAt        *time.Time
	SourceRoot       string
	Chunker          string
	ChunkSize        int
	ChunkOverlap     int
	IndexTextVersion int
	Status           string
	Error            *string
}

type Chunk struct {
	ID             string
	DocumentID     string
	ChunkIndex     int
	ContentHash    string
	ChunkText      string
	IndexedText    string
	StartLine      int
	EndLine        int
	VectorID       *int64
	EmbeddingModel string
	Active         bool
	CreatedAt      time.Time
	Chunker        string
	Language       string
	HeadingPath    []string
	SymbolName     string
	SymbolKind     string
}

type Stats struct {
	IndexedDocuments int
	ErrorDocuments   int
	DeletedDocuments int
	ActiveChunks     int
	InactiveChunks   int
	LastIndexedAt    *time.Time
}

type PendingRun struct {
	ID            string
	DocumentCount int
	ChunkCount    int
	CreatedAt     time.Time
}

type ChunkVectorUpdate struct {
	ChunkID        string
	VectorID       int64
	EmbeddingModel string
}

type LexicalHit struct {
	ChunkID string
	Score   float64
}
