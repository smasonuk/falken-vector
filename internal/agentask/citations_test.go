package agentask

import (
	"testing"

	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
)

func TestCitationRegistryFirstRegistrationAssignsSourceOne(t *testing.T) {
	registry := NewCitationRegistry()
	source := registry.Register(testRetrievedChunk("chunk-1", "README.md", "raw text", "indexed text"))
	if source.SourceNumber != 1 {
		t.Fatalf("source number = %d, want 1", source.SourceNumber)
	}
}

func TestCitationRegistryDuplicateChunkIDReusesSourceNumber(t *testing.T) {
	registry := NewCitationRegistry()
	first := registry.Register(testRetrievedChunk("chunk-1", "README.md", "raw text", "indexed text"))
	second := registry.Register(testRetrievedChunk("chunk-1", "README.md", "changed text", "changed indexed"))
	if first.SourceNumber != second.SourceNumber {
		t.Fatalf("duplicate source numbers = %d and %d, want same", first.SourceNumber, second.SourceNumber)
	}
	if registry.SourceCount() != 1 {
		t.Fatalf("source count = %d, want 1", registry.SourceCount())
	}
}

func TestCitationRegistryEmptyChunkIDDedupesByFallbackKey(t *testing.T) {
	registry := NewCitationRegistry()
	first := registry.Register(testRetrievedChunk("", "README.md", "raw text", "indexed text"))
	second := registry.Register(testRetrievedChunk("", "README.md", "raw text", "other indexed"))
	third := registry.Register(testRetrievedChunk("", "README.md", "different text", "indexed text"))
	if first.SourceNumber != second.SourceNumber {
		t.Fatalf("fallback duplicate source numbers = %d and %d, want same", first.SourceNumber, second.SourceNumber)
	}
	if third.SourceNumber == first.SourceNumber {
		t.Fatalf("different fallback text reused source %d", third.SourceNumber)
	}
}

func TestCitationRegistrySourcesSortedBySourceNumber(t *testing.T) {
	registry := NewCitationRegistry()
	registry.Register(testRetrievedChunk("chunk-1", "one.md", "one", "one"))
	registry.Register(testRetrievedChunk("chunk-2", "two.md", "two", "two"))
	sources := registry.Sources()
	if len(sources) != 2 || sources[0].SourceNumber != 1 || sources[1].SourceNumber != 2 {
		t.Fatalf("sources = %+v, want sorted by source number", sources)
	}
}

func TestCitationRegistryValidateReusesRAGValidator(t *testing.T) {
	registry := NewCitationRegistry()
	registry.Register(testRetrievedChunk("chunk-1", "README.md", "raw text", "indexed text"))
	validation := registry.Validate("answer [source 2]")
	if validation.Valid || len(validation.OutOfRangeSources) != 1 || validation.OutOfRangeSources[0] != 2 {
		t.Fatalf("validation = %+v, want out-of-range source 2", validation)
	}
}

func TestCitationRegistryUsesChunkTextNotIndexedText(t *testing.T) {
	registry := NewCitationRegistry()
	source := registry.Register(testRetrievedChunk("chunk-1", "README.md", "raw chunk text", "Document: hidden"))
	if source.Text != "raw chunk text" {
		t.Fatalf("source text = %q, want raw chunk text", source.Text)
	}
}

func testRetrievedChunk(id, path, chunkText, indexedText string) rag.RetrievedChunk {
	return rag.RetrievedChunk{
		Path:  path,
		Score: 0.5,
		Chunk: manifest.Chunk{
			ID:          id,
			DocumentID:  "doc",
			ChunkText:   chunkText,
			IndexedText: indexedText,
			StartLine:   10,
			EndLine:     12,
		},
	}
}
