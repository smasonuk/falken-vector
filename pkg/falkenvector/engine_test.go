package falkenvector

import (
	"context"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	internalconfig "github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/manifest"
)

func TestNewEngineMinimalLexicalCloseAndSilent(t *testing.T) {
	stateDir := t.TempDir()
	stdout, stderr := captureProcessOutput(t, func() {
		engine, err := NewEngine(EngineConfig{
			StateDir:  stateDir,
			Retrieval: RetrievalOptions{Mode: RetrievalLexical},
		})
		if err != nil {
			t.Fatalf("NewEngine: %v", err)
		}
		if err := engine.Close(); err != nil {
			t.Fatalf("Close first: %v", err)
		}
		if err := engine.Close(); err != nil {
			t.Fatalf("Close second: %v", err)
		}
	})
	if stdout != "" || stderr != "" {
		t.Fatalf("stdout=%q stderr=%q, want silent", stdout, stderr)
	}
}

func TestEngineConfigFromEnvReadsFalkengoKeys(t *testing.T) {
	values := map[string]string{
		"FALKENGO_STATE_DIR":               "/tmp/falken-state",
		"FALKENGO_EMBEDDING_MODEL_API_KEY": "embed-key",
		"FALKENGO_EMBEDDING_MODEL":         "embed-model",
		"FALKENGO_EMBEDDING_MODEL_URL":     "https://embed.test/v1",
		"FALKENGO_LLM_API_KEY":             "chat-key",
		"FALKENGO_LLM_BASE_URL":            "https://chat.test/v1",
		"FALKENGO_LLM_MODEL":               "chat-model",
	}

	config := EngineConfigFromEnv(func(key string) string {
		return values[key]
	})

	if config.StateDir != "/tmp/falken-state" {
		t.Fatalf("StateDir = %q", config.StateDir)
	}
	if config.Embedding.APIKey != "embed-key" || config.Embedding.BaseURL != "https://embed.test/v1" || config.Embedding.Model != "embed-model" {
		t.Fatalf("Embedding = %+v", config.Embedding)
	}
	if config.Chat.APIKey != "chat-key" || config.Chat.BaseURL != "https://chat.test/v1" || config.Chat.Model != "chat-model" {
		t.Fatalf("Chat = %+v", config.Chat)
	}
}

func TestEngineConfigFromEnvFallsBackToEmbeddingKeyForChat(t *testing.T) {
	config := EngineConfigFromEnv(func(key string) string {
		if key == "FALKENGO_EMBEDDING_MODEL_API_KEY" {
			return "shared-key"
		}
		return ""
	})
	if config.Chat.APIKey != "shared-key" {
		t.Fatalf("Chat APIKey = %q, want fallback embedding key", config.Chat.APIKey)
	}
}

func TestEngineQueryLexicalReturnsChunksAndEvents(t *testing.T) {
	stateDir := t.TempDir()
	seedEngineManifest(t, stateDir)
	var events []Event
	engine, err := NewEngine(EngineConfig{
		StateDir: stateDir,
		Retrieval: RetrievalOptions{
			Mode: RetrievalLexical,
			TopK: 3,
		},
		Events: func(event Event) {
			events = append(events, event)
		},
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	result, err := engine.Query(context.Background(), QueryRequest{Question: "AlphaToken"})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if result.Question != "AlphaToken" || len(result.Chunks) != 1 {
		t.Fatalf("result = %+v, want one chunk", result)
	}
	if !strings.Contains(result.Chunks[0].Text, "AlphaToken") {
		t.Fatalf("chunk text = %q, want raw result text", result.Chunks[0].Text)
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, []EventType{EventRetrievalStarted, EventRetrievalCompleted}) {
		t.Fatalf("events = %v", got)
	}
	if events[0].RunID == "" || events[0].Seq != 1 || events[1].Seq != 2 || events[0].At.IsZero() {
		t.Fatalf("event metadata = %+v %+v", events[0], events[1])
	}
	if events[1].Retrieval == nil || events[1].Retrieval.ResultCount != 1 {
		t.Fatalf("completed retrieval event = %+v", events[1].Retrieval)
	}
	if len(events[1].Retrieval.Sources) != 1 || events[1].Retrieval.Sources[0].Text != "" {
		t.Fatalf("retrieval sources = %+v, want redacted event text by default", events[1].Retrieval.Sources)
	}
}

func TestEngineAskRAGReturnsAnswerAndEventOrder(t *testing.T) {
	stateDir := t.TempDir()
	seedEngineManifest(t, stateDir)
	var events []Event
	engine, err := NewEngine(EngineConfig{
		StateDir: stateDir,
		Retrieval: RetrievalOptions{
			Mode: RetrievalLexical,
			TopK: 1,
		},
		Chat: ModelConfig{
			APIKey:  "chat-key",
			BaseURL: "https://chat.test/v1",
			Model:   "chat-model",
		},
		HTTPClient: fakeHTTPClient{body: `{"model":"chat-model","choices":[{"message":{"role":"assistant","content":"Alpha is documented. [source 1]"}}]}`},
		Events: func(event Event) {
			events = append(events, event)
		},
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	answer, err := engine.Ask(context.Background(), AskRequest{Question: "AlphaToken"})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if answer.Text != "Alpha is documented. [source 1]" || len(answer.Sources) != 1 || !answer.CitationValid {
		t.Fatalf("answer = %+v", answer)
	}
	want := []EventType{
		EventRunStarted,
		EventRetrievalStarted,
		EventRetrievalCompleted,
		EventLLMRequest,
		EventLLMResponse,
		EventRunCompleted,
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
}

func TestEngineAskNoChunksIsSilentStructuredResult(t *testing.T) {
	stateDir := t.TempDir()
	seedEngineManifest(t, stateDir)
	engine, err := NewEngine(EngineConfig{
		StateDir:  stateDir,
		Retrieval: RetrievalOptions{Mode: RetrievalLexical},
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var answer Answer
	stdout, stderr := captureProcessOutput(t, func() {
		var err error
		answer, err = engine.Ask(context.Background(), AskRequest{Question: "NoSuchNeedle"})
		if err != nil {
			t.Fatalf("Ask: %v", err)
		}
	})
	if stdout != "" || stderr != "" {
		t.Fatalf("stdout=%q stderr=%q, want silent", stdout, stderr)
	}
	if answer.Text != "" || len(answer.Sources) != 0 || !answer.CitationValid {
		t.Fatalf("answer = %+v, want structured no-evidence result", answer)
	}
}

func TestEngineStatusReturnsPathsStatsAndIsSilent(t *testing.T) {
	stateDir := t.TempDir()
	seedEngineManifest(t, stateDir)
	engine, err := NewEngine(EngineConfig{StateDir: stateDir})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var status Status
	stdout, stderr := captureProcessOutput(t, func() {
		var err error
		status, err = engine.Status(context.Background())
		if err != nil {
			t.Fatalf("Status: %v", err)
		}
	})
	if stdout != "" || stderr != "" {
		t.Fatalf("stdout=%q stderr=%q, want silent", stdout, stderr)
	}
	if status.StateDir == "" || status.ManifestPath == "" || status.VectorPath == "" {
		t.Fatalf("status paths = %+v", status)
	}
	if status.Documents.Indexed != 1 || status.Chunks.Active != 1 || status.LastIndexedAt == nil {
		t.Fatalf("status = %+v, want seeded stats", status)
	}
}

func TestEngineStatusMissingManifestErrorsClearly(t *testing.T) {
	engine, err := NewEngine(EngineConfig{StateDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	_, err = engine.Status(context.Background())
	if err == nil || !strings.Contains(err.Error(), "No index found") {
		t.Fatalf("Status error = %v, want no-index error", err)
	}
}

func TestEngineCompactDryRunReturnsSummaryEventsAndIsSilent(t *testing.T) {
	stateDir := t.TempDir()
	seedEngineManifest(t, stateDir)
	var events []Event
	engine, err := NewEngine(EngineConfig{
		StateDir: stateDir,
		Events: func(event Event) {
			events = append(events, event)
		},
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	var result CompactResult
	stdout, stderr := captureProcessOutput(t, func() {
		var err error
		result, err = engine.Compact(context.Background(), CompactRequest{DryRun: true})
		if err != nil {
			t.Fatalf("Compact: %v", err)
		}
	})
	if stdout != "" || stderr != "" {
		t.Fatalf("stdout=%q stderr=%q, want silent", stdout, stderr)
	}
	if result.ActiveChunks != 1 || result.ReembeddedChunks != 0 || result.UpdatedChunks != 0 {
		t.Fatalf("compact result = %+v, want dry-run summary", result)
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, []EventType{EventRunStarted, EventRunCompleted}) {
		t.Fatalf("events = %v", got)
	}
}

func seedEngineManifest(t *testing.T, stateDir string) {
	t.Helper()
	ctx := context.Background()
	paths, err := internalconfig.ResolvePaths(stateDir)
	if err != nil {
		t.Fatalf("ResolvePaths: %v", err)
	}
	store, err := manifest.Open(paths.ManifestPath)
	if err != nil {
		t.Fatalf("Open manifest: %v", err)
	}
	defer store.Close()
	if err := store.Init(ctx); err != nil {
		t.Fatalf("Init manifest: %v", err)
	}
	now := time.Now().UTC()
	docID := manifest.DocumentID("/repo/docs/alpha.md")
	chunkID := manifest.ChunkID(docID, 0, "alpha")
	if err := store.ReplaceDocumentChunks(ctx, manifest.Document{
		ID:          docID,
		Path:        "/repo/docs/alpha.md",
		ContentHash: "hash",
		SizeBytes:   42,
		ModifiedAt:  now,
		IndexedAt:   &now,
		Status:      manifest.DocumentStatusIndexed,
	}, []manifest.Chunk{{
		ID:             chunkID,
		DocumentID:     docID,
		ChunkIndex:     0,
		ContentHash:    "alpha",
		ChunkText:      "AlphaToken appears in the indexed documentation.",
		IndexedText:    "Document: docs/alpha.md\nChunk:\nAlphaToken appears in the indexed documentation.",
		StartLine:      3,
		EndLine:        5,
		EmbeddingModel: "embed-model",
		Active:         true,
		CreatedAt:      now,
	}}); err != nil {
		t.Fatalf("ReplaceDocumentChunks: %v", err)
	}
}

type fakeHTTPClient struct {
	body string
	err  error
}

func (f fakeHTTPClient) Do(*http.Request) (*http.Response, error) {
	if f.err != nil {
		return nil, f.err
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(f.body)),
	}, nil
}

func captureProcessOutput(t *testing.T, fn func()) (string, string) {
	t.Helper()
	oldStdout := os.Stdout
	oldStderr := os.Stderr
	stdoutReader, stdoutWriter, err := os.Pipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	stderrReader, stderrWriter, err := os.Pipe()
	if err != nil {
		t.Fatalf("stderr pipe: %v", err)
	}
	os.Stdout = stdoutWriter
	os.Stderr = stderrWriter
	defer func() {
		os.Stdout = oldStdout
		os.Stderr = oldStderr
	}()

	fn()

	if err := stdoutWriter.Close(); err != nil {
		t.Fatalf("close stdout writer: %v", err)
	}
	if err := stderrWriter.Close(); err != nil {
		t.Fatalf("close stderr writer: %v", err)
	}
	stdoutBytes, err := io.ReadAll(stdoutReader)
	if err != nil {
		t.Fatalf("read stdout: %v", err)
	}
	stderrBytes, err := io.ReadAll(stderrReader)
	if err != nil {
		t.Fatalf("read stderr: %v", err)
	}
	return string(stdoutBytes), string(stderrBytes)
}
