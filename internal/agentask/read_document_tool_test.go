package agentask

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
)

func TestReadIndexDocumentToolUnknownSourceFails(t *testing.T) {
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: NewCitationRegistry()})
	result := executeReadDocumentTool(t, tool, `{"source_number":1}`)
	if result.Success || result.Status != "unknown_source" {
		t.Fatalf("result = %+v, want unknown source failure", result)
	}
}

func TestReadIndexDocumentToolRejectsArbitraryPathArgument(t *testing.T) {
	registry := NewCitationRegistry()
	registry.Register(testRetrievedChunk("chunk-1", "README.md", "text", "hidden"))
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: registry})
	result := executeReadDocumentTool(t, tool, `{"source_number":1,"path":"secret.txt"}`)
	if result.Success || result.Status != "invalid_arguments" {
		t.Fatalf("result = %+v, want strict argument failure", result)
	}
}

func TestReadIndexDocumentToolReadsWholeSmallFile(t *testing.T) {
	file := writeReadSourceFileNamed(t, "small.md", "one\ntwo\nthree\n")
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 2, 2))
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: registry})
	result := executeReadDocumentTool(t, tool, `{"source_number":1}`)
	if !result.Success || result.Status != "ok" {
		t.Fatalf("result = %+v, want success", result)
	}
	if !strings.Contains(result.Content, "one\ntwo\nthree") || !strings.Contains(result.Content, "Continue citing [source 1].") {
		t.Fatalf("content = %q, want whole file context", result.Content)
	}
	source, _ := registry.SourceByNumber(1)
	if source.StartLine != 1 || source.EndLine != 3 {
		t.Fatalf("source = %+v, want expanded whole file", source)
	}
	payload := decodeReadDocumentPayload(t, result.Payload)
	if payload.Mode != "whole" || payload.Lines != 3 || payload.EstimatedTokens == 0 || payload.FileLineCount != 3 {
		t.Fatalf("payload = %+v, want whole-file stats", payload)
	}
}

func TestReadIndexDocumentToolReadsRange(t *testing.T) {
	file := writeReadSourceFileNamed(t, "range.md", numberedLines(10))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 5, 5))
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: registry})
	result := executeReadDocumentTool(t, tool, `{"source_number":1,"mode":"range","start_line":3,"end_line":6}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if strings.Contains(result.Content, "line 2") || !strings.Contains(result.Content, "line 3") || !strings.Contains(result.Content, "line 6") {
		t.Fatalf("content = %q, want clamped requested range", result.Content)
	}
	payload := decodeReadDocumentPayload(t, result.Payload)
	if payload.Mode != "range" || payload.StartLine != 3 || payload.EndLine != 6 {
		t.Fatalf("payload = %+v, want range 3-6", payload)
	}
}

func TestReadIndexDocumentToolMaxLinesTooLargeOmitsText(t *testing.T) {
	file := writeReadSourceFileNamed(t, "big.md", numberedLines(20))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 1, 1))
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: registry, MaxLines: 5})
	result := executeReadDocumentTool(t, tool, `{"source_number":1}`)
	if !result.Success || result.Status != "too_large" {
		t.Fatalf("result = %+v, want too_large success", result)
	}
	if strings.Contains(result.Content, "line 20") {
		t.Fatalf("content = %q, leaked large file text", result.Content)
	}
	payload := decodeReadDocumentPayload(t, result.Payload)
	if payload.Text != "" || payload.Lines != 20 || payload.MaxLines != 5 {
		t.Fatalf("payload = %+v, want no text and line cap stats", payload)
	}
}

func TestReadIndexDocumentToolMaxTokensTooLargeOmitsText(t *testing.T) {
	file := writeReadSourceFileNamed(t, "tokens.md", strings.Repeat("token heavy line\n", 20))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 1, 1))
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: registry, MaxTokens: 5})
	result := executeReadDocumentTool(t, tool, `{"source_number":1}`)
	payload := decodeReadDocumentPayload(t, result.Payload)
	if !result.Success || result.Status != "too_large" || payload.Text != "" || payload.EstimatedTokens <= payload.MaxTokens {
		t.Fatalf("result = %+v payload = %+v, want token cap too_large without text", result, payload)
	}
}

func TestReadIndexDocumentToolMissingFileFailsCleanly(t *testing.T) {
	registry := NewCitationRegistry()
	missing := writeReadSourceFileNamed(t, "exists.md", "text") + ".missing"
	registry.Register(testRetrievedChunk("chunk-1", missing, "text", "hidden"))
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: registry})
	result := executeReadDocumentTool(t, tool, `{"source_number":1}`)
	if result.Success || result.Status != "read_document_failed" {
		t.Fatalf("result = %+v, want missing file failure", result)
	}
}

func TestReadIndexDocumentToolParentMarkdownSection(t *testing.T) {
	file := writeReadSourceFileNamed(t, "doc.md", "# Intro\nno\n# AlphaFold\none\n## Detail\ntwo\n# Later\nthree\n")
	registry := NewCitationRegistry()
	chunk := readSourceRetrievedChunk(file, 6, 6)
	chunk.Chunk.Chunker = "markdown"
	chunk.Chunk.HeadingPath = []string{"AlphaFold", "Detail"}
	registry.Register(chunk)
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: registry})
	result := executeReadDocumentTool(t, tool, `{"source_number":1,"mode":"parent"}`)
	payload := decodeReadDocumentPayload(t, result.Payload)
	if !result.Success || payload.ParentKind != string(ParentContextMarkdownHeadingSection) || payload.StartLine != 5 || payload.EndLine != 6 {
		t.Fatalf("result = %+v payload = %+v, want nested markdown heading section", result, payload)
	}
}

func TestReadIndexDocumentToolParentCodeSymbol(t *testing.T) {
	file := writeReadSourceFileNamed(t, "code.go", "package main\n\nfunc One() {\n println(1)\n}\n\nfunc Two() {\n println(2)\n}\n")
	registry := NewCitationRegistry()
	chunk := readSourceRetrievedChunk(file, 4, 4)
	chunk.Chunk.Chunker = "code"
	chunk.Chunk.Language = "go"
	chunk.Chunk.SymbolName = "One"
	chunk.Chunk.SymbolKind = "function"
	registry.Register(chunk)
	tool := NewReadIndexDocumentTool(ReadDocumentToolOptions{Registry: registry})
	result := executeReadDocumentTool(t, tool, `{"source_number":1,"mode":"parent"}`)
	payload := decodeReadDocumentPayload(t, result.Payload)
	if !result.Success || payload.ParentKind != string(ParentContextCodeSymbolSection) || payload.StartLine != 3 || payload.EndLine != 6 {
		t.Fatalf("result = %+v payload = %+v, want function section", result, payload)
	}
	if strings.Contains(result.Content, "func Two") {
		t.Fatalf("content = %q, should not include next function", result.Content)
	}
}

func executeReadDocumentTool(t *testing.T, tool falken.Tool, args string) falken.ToolExecutionResult {
	t.Helper()
	result, err := tool.Execute(context.Background(), falken.ToolInvocation{
		CallID:    "call-read-doc",
		Name:      ReadIndexDocumentToolName,
		Arguments: json.RawMessage(args),
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	return result
}

func decodeReadDocumentPayload(t *testing.T, raw []byte) readDocumentPayload {
	t.Helper()
	var payload readDocumentPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("decode payload: %v\n%s", err, string(raw))
	}
	return payload
}
