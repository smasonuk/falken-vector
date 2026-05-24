package agentask

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"sync"

	"github.com/smasonuk/falken-vector/internal/rag"
)

type CitationRegistry struct {
	mu       sync.Mutex
	byKey    map[string]rag.SourceChunk
	byNumber map[int]rag.SourceChunk
	next     int
}

func NewCitationRegistry() *CitationRegistry {
	return &CitationRegistry{
		byKey:    make(map[string]rag.SourceChunk),
		byNumber: make(map[int]rag.SourceChunk),
		next:     1,
	}
}

func (r *CitationRegistry) Register(chunk rag.RetrievedChunk) rag.SourceChunk {
	if r == nil {
		r = NewCitationRegistry()
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	key := citationKey(chunk)
	if source, ok := r.byKey[key]; ok {
		return source
	}
	source := rag.SourceChunk{
		SourceNumber: r.next,
		Path:         chunk.Path,
		StartLine:    chunk.Chunk.StartLine,
		EndLine:      chunk.Chunk.EndLine,
		Text:         chunk.Chunk.ChunkText,
		Score:        chunk.Score,
	}
	r.next++
	r.byKey[key] = source
	r.byNumber[source.SourceNumber] = source
	return source
}

func (r *CitationRegistry) Sources() []rag.SourceChunk {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	numbers := make([]int, 0, len(r.byNumber))
	for number := range r.byNumber {
		numbers = append(numbers, number)
	}
	sort.Ints(numbers)
	sources := make([]rag.SourceChunk, 0, len(numbers))
	for _, number := range numbers {
		sources = append(sources, r.byNumber[number])
	}
	return sources
}

func (r *CitationRegistry) SourceByNumber(n int) (rag.SourceChunk, bool) {
	if r == nil {
		return rag.SourceChunk{}, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	source, ok := r.byNumber[n]
	return source, ok
}

func (r *CitationRegistry) ExpandSource(sourceNumber int, startLine int, endLine int, text string) error {
	if r == nil {
		return fmt.Errorf("citation registry is nil")
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	source, ok := r.byNumber[sourceNumber]
	if !ok {
		return fmt.Errorf("unknown source [source %d]", sourceNumber)
	}
	source.StartLine = startLine
	source.EndLine = endLine
	source.Text = text
	r.byNumber[sourceNumber] = source
	for key, existing := range r.byKey {
		if existing.SourceNumber == sourceNumber {
			r.byKey[key] = source
			break
		}
	}
	return nil
}

func (r *CitationRegistry) SourceCount() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.byNumber)
}

func (r *CitationRegistry) Validate(answer string) rag.CitationValidation {
	return rag.ValidateAnswerCitations(answer, r.SourceCount())
}

func citationKey(chunk rag.RetrievedChunk) string {
	if chunk.Chunk.ID != "" {
		return "chunk:" + chunk.Chunk.ID
	}
	sum := sha256.Sum256([]byte(chunk.Chunk.ChunkText))
	return fmt.Sprintf("fallback:%s:%d:%d:%s", chunk.Path, chunk.Chunk.StartLine, chunk.Chunk.EndLine, hex.EncodeToString(sum[:]))
}
