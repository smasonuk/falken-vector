package ingest

import (
	"strings"
	"testing"
)

func TestChunkTextShortFile(t *testing.T) {
	chunks, err := ChunkText("hello\nworld", ChunkOptions{ChunkSize: 1200, ChunkOverlap: 200})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks = %d, want 1", len(chunks))
	}
	if chunks[0].Text != "hello\nworld" || chunks[0].StartLine != 1 || chunks[0].EndLine != 2 {
		t.Fatalf("chunk = %+v", chunks[0])
	}
}

func TestChunkTextLongFileOverlap(t *testing.T) {
	input := strings.Repeat("a", 10)
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 4, ChunkOverlap: 1, Mode: ChunkerModeFixed})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d, want 3", len(chunks))
	}
	if chunks[0].Text != "aaaa" || chunks[1].Text != "aaaa" || chunks[2].Text != "aaaa" {
		t.Fatalf("chunks = %+v", chunks)
	}
}

func TestChunkTextLineNumbers(t *testing.T) {
	chunks, err := ChunkText("one\ntwo\nthree\nfour", ChunkOptions{ChunkSize: 8, ChunkOverlap: 0, Mode: ChunkerModeFixed})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if chunks[0].StartLine != 1 || chunks[0].EndLine != 3 {
		t.Fatalf("first chunk lines = %d-%d", chunks[0].StartLine, chunks[0].EndLine)
	}
	if chunks[1].StartLine != 3 || chunks[1].EndLine != 4 {
		t.Fatalf("second chunk lines = %d-%d", chunks[1].StartLine, chunks[1].EndLine)
	}
}

func TestParseAndDetectChunkerMode(t *testing.T) {
	for _, value := range []string{"", "auto", "fixed", "markdown", "text", "code"} {
		if _, err := ParseChunkerMode(value); err != nil {
			t.Fatalf("ParseChunkerMode(%q): %v", value, err)
		}
	}
	if _, err := ParseChunkerMode("bogus"); err == nil {
		t.Fatal("ParseChunkerMode(bogus) succeeded, want error")
	}
	tests := map[string]ChunkerMode{
		"README.md":              ChunkerModeMarkdown,
		"notes.txt":              ChunkerModeText,
		"internal/ingest/foo.go": ChunkerModeCode,
		"unknown.xyz":            ChunkerModeText,
	}
	for path, want := range tests {
		if got := DetectChunkerMode(path); got != want {
			t.Fatalf("DetectChunkerMode(%q) = %q, want %q", path, got, want)
		}
	}
}

func TestChunkMarkdownSplitsByHeadingsAndPreservesPath(t *testing.T) {
	input := "# Project\n\nIntro.\n\n## Install\n\nRun this.\n\n## Usage\n\nDo that.\n"
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 1200, ChunkOverlap: 0, Mode: ChunkerModeMarkdown, Path: "README.md"})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d, want 3: %+v", len(chunks), chunks)
	}
	if !sameStrings(chunks[0].HeadingPath, []string{"Project"}) || !sameStrings(chunks[1].HeadingPath, []string{"Project", "Install"}) || !sameStrings(chunks[2].HeadingPath, []string{"Project", "Usage"}) {
		t.Fatalf("heading paths = %+v, %+v, %+v", chunks[0].HeadingPath, chunks[1].HeadingPath, chunks[2].HeadingPath)
	}
}

func TestChunkMarkdownDoesNotSplitFencedCodeBlock(t *testing.T) {
	input := "# Project\n\n```md\n# Not a heading\n```\n\n## Usage\n\nDo that.\n"
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 1200, ChunkOverlap: 0, Mode: ChunkerModeMarkdown, Path: "README.md"})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2: %+v", len(chunks), chunks)
	}
	if !strings.Contains(chunks[0].Text, "# Not a heading") || !sameStrings(chunks[0].HeadingPath, []string{"Project"}) {
		t.Fatalf("first chunk = %+v", chunks[0])
	}
	if !sameStrings(chunks[1].HeadingPath, []string{"Project", "Usage"}) {
		t.Fatalf("second heading path = %+v", chunks[1].HeadingPath)
	}
}

func TestChunkPlainTextPacksParagraphs(t *testing.T) {
	input := "alpha paragraph\n\nbeta paragraph\n\ngamma paragraph\n"
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 40, ChunkOverlap: 0, Mode: ChunkerModeText})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2: %+v", len(chunks), chunks)
	}
	if !strings.Contains(chunks[0].Text, "alpha") || !strings.Contains(chunks[0].Text, "beta") || !strings.Contains(chunks[1].Text, "gamma") {
		t.Fatalf("chunks = %+v", chunks)
	}
}

func TestChunkPlainTextSplitsLargeParagraphWithFallback(t *testing.T) {
	input := strings.Repeat("a", 25)
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 10, ChunkOverlap: 2, Mode: ChunkerModeText})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d, want 3: %+v", len(chunks), chunks)
	}
	for _, chunk := range chunks {
		if chunk.Chunker != string(ChunkerModeText) {
			t.Fatalf("chunker = %q, want text", chunk.Chunker)
		}
	}
}

func TestChunkPlainTextOverlapDoesNotEmitDuplicateWhenCarryDoesNotFit(t *testing.T) {
	input := "aaaaaaaa\n\nbbbbbbbb\n\ncccccccccccccc\n"
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 20, ChunkOverlap: 1, Mode: ChunkerModeText})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want 2: %+v", len(chunks), chunks)
	}
	if chunks[0].Text != "aaaaaaaa\n\nbbbbbbbb" || chunks[1].Text != "cccccccccccccc" {
		t.Fatalf("chunks = %+v", chunks)
	}
}

func TestChunkCodeDoesNotSplitIndentedPythonMethods(t *testing.T) {
	input := "class Thing:\n    def run(self):\n        pass\n\ndef build():\n    pass\n"
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 1200, ChunkOverlap: 0, Mode: ChunkerModeCode, Path: "demo.py"})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks = %d, want class + top-level function: %+v", len(chunks), chunks)
	}
	if chunks[0].SymbolName != "Thing" || !strings.Contains(chunks[0].Text, "def run") {
		t.Fatalf("class chunk = %+v", chunks[0])
	}
	if chunks[1].SymbolName != "build" {
		t.Fatalf("function chunk = %+v", chunks[1])
	}
}

func TestChunkCodeDetectsGoFunctionsAndTypes(t *testing.T) {
	input := "package demo\n\n// Person is a model.\ntype Person struct{}\n\n// Run starts work.\nfunc Run() {}\n"
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 1200, ChunkOverlap: 0, Mode: ChunkerModeCode, Path: "demo.go"})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d, want preamble + type + func: %+v", len(chunks), chunks)
	}
	if chunks[1].Language != "go" || chunks[1].SymbolName != "Person" || chunks[1].SymbolKind != "type" {
		t.Fatalf("type chunk = %+v", chunks[1])
	}
	if chunks[2].SymbolName != "Run" || chunks[2].SymbolKind != "function" || !strings.Contains(chunks[2].Text, "// Run starts work.") {
		t.Fatalf("func chunk = %+v", chunks[2])
	}
}

func TestChunkCodeDetectsPythonFunctionsAndClasses(t *testing.T) {
	input := "import os\n\nclass Thing:\n    pass\n\ndef run():\n    pass\n"
	chunks, err := ChunkText(input, ChunkOptions{ChunkSize: 1200, ChunkOverlap: 0, Mode: ChunkerModeCode, Path: "demo.py"})
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d, want preamble + class + function: %+v", len(chunks), chunks)
	}
	if chunks[1].Language != "python" || chunks[1].SymbolName != "Thing" || chunks[1].SymbolKind != "class" {
		t.Fatalf("class chunk = %+v", chunks[1])
	}
	if chunks[2].SymbolName != "run" || chunks[2].SymbolKind != "function" {
		t.Fatalf("function chunk = %+v", chunks[2])
	}
}

func sameStrings(got []string, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestChunkTextInvalidOptions(t *testing.T) {
	tests := []ChunkOptions{
		{ChunkSize: 0, ChunkOverlap: 0},
		{ChunkSize: 10, ChunkOverlap: -1},
		{ChunkSize: 10, ChunkOverlap: 10},
	}
	for _, opts := range tests {
		if _, err := ChunkText("text", opts); err == nil {
			t.Fatalf("ChunkText(%+v) succeeded, want error", opts)
		}
	}
}
