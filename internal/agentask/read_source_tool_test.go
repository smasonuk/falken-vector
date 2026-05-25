package agentask

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/smasonuk/falken-core/pkg/falken"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
)

func TestReadIndexSourceToolUnknownSourceFails(t *testing.T) {
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: NewCitationRegistry()})
	result := executeReadSourceTool(t, tool, `{"source_number":1}`)
	if result.Success || result.Status != "unknown_source" {
		t.Fatalf("result = %+v, want unknown source failure", result)
	}
}

func TestReadIndexSourceToolReadsNearbyContext(t *testing.T) {
	file := writeReadSourceFile(t, "one\ntwo\nthree\nfour\nfive\n")
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 3, 3))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})

	result := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":1}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if !strings.Contains(result.Content, "[source 1]") || !strings.Contains(result.Content, "two\nthree\nfour") {
		t.Fatalf("content = %q, want expanded source 1 context", result.Content)
	}
	payload := decodeReadSourcePayload(t, result.Payload)
	if payload.StartLine != 2 || payload.EndLine != 4 {
		t.Fatalf("payload lines = %d-%d, want nearby context around line 3", payload.StartLine, payload.EndLine)
	}
}

func TestReadIndexSourceToolCapsContextLines(t *testing.T) {
	file := writeReadSourceFile(t, numberedLines(220))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 100, 100))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})

	result := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":1000}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	payload := decodeReadSourcePayload(t, result.Payload)
	if payload.StartLine != 1 || payload.EndLine != 200 {
		t.Fatalf("payload lines = %d-%d, want context capped at 100 around line 100", payload.StartLine, payload.EndLine)
	}
	if len(payload.Warnings) != 1 || !strings.Contains(payload.Warnings[0], "capped") {
		t.Fatalf("warnings = %+v, want cap warning", payload.Warnings)
	}
}

func TestReadIndexSourceToolRejectsArbitraryPathArgument(t *testing.T) {
	registry := NewCitationRegistry()
	registry.Register(testRetrievedChunk("chunk-1", "README.md", "text", "hidden"))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})
	result := executeReadSourceTool(t, tool, `{"source_number":1,"path":"secret.txt"}`)
	if result.Success || result.Status != "invalid_arguments" {
		t.Fatalf("result = %+v, want strict argument failure", result)
	}
}

func TestReadIndexSourceToolUsesSameSourceNumber(t *testing.T) {
	file := writeReadSourceFile(t, "one\ntwo\nthree\n")
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 2, 2))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})
	result := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":0}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	if !strings.Contains(result.Content, "Continue citing [source 1].") || registry.SourceCount() != 1 {
		t.Fatalf("content = %q source count = %d, want same source number", result.Content, registry.SourceCount())
	}
}

func TestReadIndexSourceToolExpandsRegisteredSourceRange(t *testing.T) {
	file := writeReadSourceFile(t, numberedLines(40))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 10, 12))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})

	result := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":20}`)
	if !result.Success {
		t.Fatalf("result = %+v, want success", result)
	}
	source, ok := registry.SourceByNumber(1)
	if !ok {
		t.Fatal("source 1 missing after expansion")
	}
	if source.StartLine != 1 || source.EndLine != 32 {
		t.Fatalf("source lines = %d-%d, want expanded 1-32", source.StartLine, source.EndLine)
	}
	if !strings.Contains(source.Text, "line 1") || !strings.Contains(source.Text, "line 32") {
		t.Fatalf("source text = %q, want expanded context text", source.Text)
	}
	if registry.SourceCount() != 1 {
		t.Fatalf("source count = %d, want same source number", registry.SourceCount())
	}
}

func TestReadIndexSourceToolSkipsAlreadyCoveredOverlap(t *testing.T) {
	file := writeReadSourceFile(t, numberedLines(150))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 20, 25))
	registry.Register(readSourceRetrievedChunk(file, 30, 35))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})

	first := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":60}`)
	if !first.Success {
		t.Fatalf("first result = %+v, want success", first)
	}
	second := executeReadSourceTool(t, tool, `{"source_number":2,"context_lines":60}`)
	if !second.Success {
		t.Fatalf("second result = %+v, want success", second)
	}
	payload := decodeReadSourcePayload(t, second.Payload)
	if payload.Status != "already_covered" {
		t.Fatalf("status = %q, want already_covered", payload.Status)
	}
	if payload.CoveredBySourceNumber != 1 {
		t.Fatalf("covered_by_source_number = %d, want 1", payload.CoveredBySourceNumber)
	}
	if payload.Text != "" {
		t.Fatalf("payload text length = %d, want no duplicate context text", len(payload.Text))
	}
	if !strings.Contains(second.Content, "[source 2] already covered by expanded [source 1]") {
		t.Fatalf("content = %q, want already-covered guidance", second.Content)
	}
}

func TestReadIndexSourceToolAllowsNonOverlappingSameDocument(t *testing.T) {
	file := writeReadSourceFile(t, numberedLines(180))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 20, 25))
	registry.Register(readSourceRetrievedChunk(file, 140, 145))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})

	first := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":10}`)
	second := executeReadSourceTool(t, tool, `{"source_number":2,"context_lines":10}`)
	if !first.Success || !second.Success {
		t.Fatalf("first = %+v second = %+v, want both success", first, second)
	}
	payload := decodeReadSourcePayload(t, second.Payload)
	if payload.Status != "ok" {
		t.Fatalf("status = %q, want ok", payload.Status)
	}
	if payload.CoveredBySourceNumber != 0 {
		t.Fatalf("covered_by_source_number = %d, want none", payload.CoveredBySourceNumber)
	}
}

func TestReadIndexSourceToolDoesNotDedupeDifferentDocuments(t *testing.T) {
	fileA := writeReadSourceFileNamed(t, "a.md", numberedLines(150))
	fileB := writeReadSourceFileNamed(t, "b.md", numberedLines(150))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(fileA, 20, 25))
	registry.Register(readSourceRetrievedChunk(fileB, 30, 35))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})

	first := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":60}`)
	second := executeReadSourceTool(t, tool, `{"source_number":2,"context_lines":60}`)
	if !first.Success || !second.Success {
		t.Fatalf("first = %+v second = %+v, want both success", first, second)
	}
	payload := decodeReadSourcePayload(t, second.Payload)
	if payload.Status != "ok" {
		t.Fatalf("status = %q, want ok", payload.Status)
	}
}

func TestReadIndexSourceToolMergesSubstantialOverlapWhenConfigured(t *testing.T) {
	file := writeReadSourceFile(t, numberedLines(150))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 20, 25))
	registry.Register(readSourceRetrievedChunk(file, 30, 35))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{
		Registry:      registry,
		OverlapPolicy: ReadSourceOverlapMerge,
	})

	first := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":60}`)
	if !first.Success {
		t.Fatalf("first result = %+v, want success", first)
	}
	second := executeReadSourceTool(t, tool, `{"source_number":2,"context_lines":60}`)
	if !second.Success {
		t.Fatalf("second result = %+v, want success", second)
	}
	payload := decodeReadSourcePayload(t, second.Payload)
	if payload.Status != "merge_existing" {
		t.Fatalf("status = %q, want merge_existing", payload.Status)
	}
	if payload.CoveredBySourceNumber != 1 || payload.CoveredByStartLine != 1 || payload.CoveredByEndLine != 95 {
		t.Fatalf("payload = %+v, want merged source 1 lines 1-95", payload)
	}
	source, ok := registry.SourceByNumber(1)
	if !ok {
		t.Fatal("source 1 missing after merge")
	}
	if source.StartLine != 1 || source.EndLine != 95 {
		t.Fatalf("source 1 lines = %d-%d, want merged 1-95", source.StartLine, source.EndLine)
	}
	if !strings.Contains(second.Content, "Continue citing [source 1]") {
		t.Fatalf("content = %q, want cite merged source guidance", second.Content)
	}
}

func TestReadIndexSourceToolAllowPolicyPreservesOverlappingRead(t *testing.T) {
	file := writeReadSourceFile(t, numberedLines(150))
	registry := NewCitationRegistry()
	registry.Register(readSourceRetrievedChunk(file, 20, 25))
	registry.Register(readSourceRetrievedChunk(file, 30, 35))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{
		Registry:      registry,
		OverlapPolicy: ReadSourceOverlapAllow,
	})

	first := executeReadSourceTool(t, tool, `{"source_number":1,"context_lines":60}`)
	second := executeReadSourceTool(t, tool, `{"source_number":2,"context_lines":60}`)
	if !first.Success || !second.Success {
		t.Fatalf("first = %+v second = %+v, want both success", first, second)
	}
	payload := decodeReadSourcePayload(t, second.Payload)
	if payload.Status != "ok" || payload.CoveredBySourceNumber != 0 {
		t.Fatalf("payload = %+v, want normal overlapping read", payload)
	}
}

func TestReadIndexSourceToolMissingFileFailsCleanly(t *testing.T) {
	registry := NewCitationRegistry()
	registry.Register(testRetrievedChunk("chunk-1", filepath.Join(t.TempDir(), "missing.go"), "text", "hidden"))
	tool := NewReadIndexSourceTool(ReadSourceToolOptions{Registry: registry})
	result := executeReadSourceTool(t, tool, `{"source_number":1}`)
	if result.Success || result.Status != "read_source_failed" || !strings.Contains(result.Error, "missing.go") {
		t.Fatalf("result = %+v, want clean read failure", result)
	}
}

func executeReadSourceTool(t *testing.T, tool falken.Tool, args string) falken.ToolExecutionResult {
	t.Helper()
	result, err := tool.Execute(context.Background(), falken.ToolInvocation{
		CallID:    "call-read",
		Name:      ReadIndexSourceToolName,
		Arguments: json.RawMessage(args),
	})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	return result
}

func writeReadSourceFile(t *testing.T, content string) string {
	t.Helper()
	return writeReadSourceFileNamed(t, "source.go", content)
}

func writeReadSourceFileNamed(t *testing.T, name string, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func numberedLines(n int) string {
	var b strings.Builder
	for i := 1; i <= n; i++ {
		fmt.Fprintf(&b, "line %d\n", i)
	}
	return b.String()
}

func readSourceRetrievedChunk(path string, start, end int) rag.RetrievedChunk {
	return rag.RetrievedChunk{
		Path:  path,
		Score: 0.5,
		Chunk: manifest.Chunk{
			ID:        fmt.Sprintf("chunk-%d-%d", start, end),
			ChunkText: "text",
			StartLine: start,
			EndLine:   end,
		},
	}
}

func decodeReadSourcePayload(t *testing.T, raw json.RawMessage) readSourcePayload {
	t.Helper()
	var payload readSourcePayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("decode payload: %v\n%s", err, string(raw))
	}
	return payload
}
