package ingest

import (
	"strings"
	"testing"
)

func TestBuildIndexedChunkTextMarkdownIncludesDocumentAndSection(t *testing.T) {
	got := BuildIndexedChunkText("docs/setup.md", TextChunk{
		Text:        "Install dependencies.",
		Chunker:     string(ChunkerModeMarkdown),
		HeadingPath: []string{"Project", "Installation", "Environment"},
	})
	want := "Document: docs/setup.md\nSection: Project > Installation > Environment\nChunk:\nInstall dependencies."
	if got != want {
		t.Fatalf("indexed text = %q, want %q", got, want)
	}
}

func TestBuildIndexedChunkTextCodeIncludesFileLanguageAndSymbol(t *testing.T) {
	got := BuildIndexedChunkText("internal/rag/retrieve.go", TextChunk{
		Text:       "return chunks, nil",
		Chunker:    string(ChunkerModeCode),
		Language:   "go",
		SymbolName: "Retrieve",
		SymbolKind: "function",
	})
	want := "File: internal/rag/retrieve.go\nLanguage: go\nSymbol: function Retrieve\nChunk:\nreturn chunks, nil"
	if got != want {
		t.Fatalf("indexed text = %q, want %q", got, want)
	}
}

func TestBuildIndexedChunkTextOmitsEmptyMetadataLines(t *testing.T) {
	got := BuildIndexedChunkText("notes/meeting.txt", TextChunk{
		Text:    "Agenda only.",
		Chunker: string(ChunkerModeText),
	})
	if strings.Contains(got, "Language:") || strings.Contains(got, "Symbol:") || strings.Contains(got, "Section:") {
		t.Fatalf("indexed text includes empty metadata: %q", got)
	}
	want := "Document: notes/meeting.txt\nChunk:\nAgenda only."
	if got != want {
		t.Fatalf("indexed text = %q, want %q", got, want)
	}
}

func TestBuildIndexedChunkTextIncludesRawChunkTextOnceAfterChunk(t *testing.T) {
	raw := "Body text\nwith two lines."
	got := BuildIndexedChunkText("notes.txt", TextChunk{Text: raw})
	if !strings.HasSuffix(got, "Chunk:\n"+raw) {
		t.Fatalf("indexed text = %q, want raw text after Chunk", got)
	}
	if strings.Count(got, raw) != 1 {
		t.Fatalf("indexed text = %q, want raw text once", got)
	}
}
