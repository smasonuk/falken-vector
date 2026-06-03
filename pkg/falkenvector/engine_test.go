package falkenvector

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/smasonuk/falken-vector/internal/agentask"
	internalconfig "github.com/smasonuk/falken-vector/internal/config"
	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
	"github.com/smasonuk/falken-vector/pkg/embeddings"
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
		"FALKENGO_EMBEDDING_MODEL_HEADERS": `{"X-Embed-Provider":"embed-provider"}`,
		"FALKENGO_LLM_API_KEY":             "chat-key",
		"FALKENGO_LLM_BASE_URL":            "https://chat.test/v1",
		"FALKENGO_LLM_MODEL":               "chat-model",
		"FALKENGO_LLM_HEADERS":             `{"X-Chat-Provider":"chat-provider"}`,
	}

	config, err := EngineConfigFromEnvE(func(key string) string {
		return values[key]
	})
	if err != nil {
		t.Fatalf("EngineConfigFromEnvE: %v", err)
	}

	if config.StateDir != "/tmp/falken-state" {
		t.Fatalf("StateDir = %q", config.StateDir)
	}
	if config.Embedding.APIKey != "embed-key" || config.Embedding.BaseURL != "https://embed.test/v1" || config.Embedding.Model != "embed-model" {
		t.Fatalf("Embedding = %+v", config.Embedding)
	}
	if config.Embedding.Headers["X-Embed-Provider"] != "embed-provider" {
		t.Fatalf("Embedding headers = %+v", config.Embedding.Headers)
	}
	if config.Chat.APIKey != "chat-key" || config.Chat.BaseURL != "https://chat.test/v1" || config.Chat.Model != "chat-model" {
		t.Fatalf("Chat = %+v", config.Chat)
	}
	if config.Chat.Headers["X-Chat-Provider"] != "chat-provider" {
		t.Fatalf("Chat headers = %+v", config.Chat.Headers)
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

func TestAgentCoverageNudgeOption(t *testing.T) {
	defaultValue, err := agentCoverageNudgeOption(AgentCoverageDefault)
	if err != nil || defaultValue != nil {
		t.Fatalf("default = %v, %v; want nil without error", defaultValue, err)
	}
	on, err := agentCoverageNudgeOption(AgentCoverageOn)
	if err != nil || on == nil || !*on {
		t.Fatalf("on = %v, %v; want true", on, err)
	}
	off, err := agentCoverageNudgeOption(AgentCoverageOff)
	if err != nil || off == nil || *off {
		t.Fatalf("off = %v, %v; want false", off, err)
	}
	if _, err := agentCoverageNudgeOption(AgentCoveragePolicy("bogus")); err == nil {
		t.Fatal("invalid policy succeeded, want error")
	}
}

func TestReadSourceToolOption(t *testing.T) {
	tests := []struct {
		name          string
		policy        ReadSourcePolicy
		legacyEnabled bool
		question      string
		want          bool
		wantErr       bool
	}{
		{name: "default broad auto enables", question: "summarize anything related to AlphaFold", want: true},
		{name: "default narrow stays disabled", question: "Where is citation validation implemented?", want: false},
		{name: "auto broad enables", policy: ReadSourcePolicyAuto, question: "overview of AlphaFold", want: true},
		{name: "on enables", policy: ReadSourcePolicyOn, question: "Where is citation validation implemented?", want: true},
		{name: "off disables legacy bool", policy: ReadSourcePolicyOff, legacyEnabled: true, question: "summarize AlphaFold", want: false},
		{name: "legacy bool enables default", legacyEnabled: true, question: "Where is citation validation implemented?", want: true},
		{name: "invalid errors", policy: ReadSourcePolicy("bogus"), question: "hello", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := readSourceToolOption(tt.policy, tt.legacyEnabled, tt.question)
			if tt.wantErr {
				if err == nil {
					t.Fatal("readSourceToolOption succeeded, want error")
				}
				return
			}
			if err != nil || got != tt.want {
				t.Fatalf("readSourceToolOption = %t, %v; want %t, nil", got, err, tt.want)
			}
		})
	}
}

func TestReadSourceOverlapPolicyOption(t *testing.T) {
	tests := []struct {
		name     string
		policy   ReadSourceOverlapPolicy
		question string
		want     agentask.ReadSourceOverlapPolicy
		wantErr  bool
	}{
		{name: "default narrow skip", question: "Where is citation validation implemented?", want: agentask.ReadSourceOverlapSkip},
		{name: "default broad merge", question: "summarize anything related to AlphaFold", want: agentask.ReadSourceOverlapMerge},
		{name: "skip", policy: ReadSourceOverlapSkip, want: agentask.ReadSourceOverlapSkip},
		{name: "merge", policy: ReadSourceOverlapMerge, want: agentask.ReadSourceOverlapMerge},
		{name: "allow", policy: ReadSourceOverlapAllow, want: agentask.ReadSourceOverlapAllow},
		{name: "invalid", policy: ReadSourceOverlapPolicy("bogus"), wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := readSourceOverlapPolicyOption(tt.policy, tt.question)
			if tt.wantErr {
				if err == nil {
					t.Fatal("readSourceOverlapPolicyOption succeeded, want error")
				}
				return
			}
			if err != nil || got != tt.want {
				t.Fatalf("readSourceOverlapPolicyOption = %q, %v; want %q, nil", got, err, tt.want)
			}
		})
	}
}

func TestMaxMergedReadSourceLinesOption(t *testing.T) {
	tests := []struct {
		name    string
		value   int
		want    int
		wantErr bool
	}{
		{name: "default", value: 0, want: 0},
		{name: "explicit", value: 120, want: 120},
		{name: "invalid", value: -1, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := maxMergedReadSourceLinesOption(tt.value)
			if tt.wantErr {
				if err == nil {
					t.Fatal("maxMergedReadSourceLinesOption succeeded, want error")
				}
				return
			}
			if err != nil || got != tt.want {
				t.Fatalf("maxMergedReadSourceLinesOption = %d, %v; want %d, nil", got, err, tt.want)
			}
		})
	}
}

func TestEngineConfigFromEnvERejectsInvalidHeaders(t *testing.T) {
	_, err := EngineConfigFromEnvE(func(key string) string {
		if key == "FALKENGO_LLM_HEADERS" {
			return `{bad json`
		}
		return ""
	})
	if err == nil {
		t.Fatal("EngineConfigFromEnvE succeeded, want error")
	}
	if !strings.Contains(err.Error(), "FALKENGO_LLM_HEADERS") {
		t.Fatalf("error = %v, want env var name", err)
	}
}

func TestEngineModelConfigTrimsAndDoesNotDefaultProvider(t *testing.T) {
	engine, err := NewEngine(EngineConfig{
		StateDir: t.TempDir(),
		Embedding: ModelConfig{
			APIKey:  " embed-key ",
			BaseURL: " https://embed.test/v1/ ",
			Model:   " embed-model ",
			Headers: map[string]string{" X-Embed-Provider ": " embed-provider "},
		},
		Chat: ModelConfig{
			BaseURL: " https://chat.test/v1/ ",
			Model:   " chat-model ",
			Headers: map[string]string{" X-Chat-Provider ": " chat-provider "},
		},
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	embedding := engine.embeddingConfig()
	if embedding.BaseURL != "https://embed.test/v1/" || embedding.Model != "embed-model" || embedding.APIKey != " embed-key " {
		t.Fatalf("embedding config = %+v", embedding)
	}
	if !reflect.DeepEqual(embedding.Headers, map[string]string{"X-Embed-Provider": "embed-provider"}) {
		t.Fatalf("embedding headers = %+v", embedding.Headers)
	}
	chat := engine.chatConfig()
	if chat.BaseURL != "https://chat.test/v1/" || chat.Model != "chat-model" || chat.APIKey != " embed-key " {
		t.Fatalf("chat config = %+v", chat)
	}
	if !reflect.DeepEqual(chat.Headers, map[string]string{"X-Chat-Provider": "chat-provider"}) {
		t.Fatalf("chat headers = %+v", chat.Headers)
	}
}

func TestEngineNewEmbedderRequiresExplicitURL(t *testing.T) {
	engine, err := NewEngine(EngineConfig{StateDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	_, err = engine.newEmbedder()
	if !errors.Is(err, embeddings.ErrBaseURLRequired) {
		t.Fatalf("newEmbedder error = %v, want base URL required", err)
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

func TestEngineListIndexedDocuments(t *testing.T) {
	stateDir := t.TempDir()
	seedEngineManifest(t, stateDir)
	engine, err := NewEngine(EngineConfig{StateDir: stateDir})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	docs, err := engine.ListIndexedDocuments(context.Background())
	if err != nil {
		t.Fatalf("ListIndexedDocuments: %v", err)
	}
	if len(docs) != 1 {
		t.Fatalf("docs = %+v, want one", docs)
	}
	if docs[0].Path != "/repo/docs/alpha.md" || docs[0].SourceRoot != "/repo" || docs[0].SizeBytes != 42 {
		t.Fatalf("doc = %+v", docs[0])
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

func TestEngineAskAttachedDocumentsBypassesRetrievalAndAgentTools(t *testing.T) {
	var events []Event
	engine, err := NewEngine(EngineConfig{
		StateDir: t.TempDir(),
		Chat: ModelConfig{
			APIKey:  "chat-key",
			BaseURL: "https://chat.test/v1",
			Model:   "chat-model",
		},
		HTTPClient: fakeHTTPClient{body: `{"model":"chat-model","choices":[{"message":{"role":"assistant","content":"Beta is in the attached document. [source 2]"}}]}`},
		Events: func(event Event) {
			events = append(events, event)
		},
	})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}

	answer, err := engine.Ask(context.Background(), AskRequest{
		Question: "Where is beta?",
		Agent:    true,
		AttachedDocuments: []AttachedDocument{
			{Path: "/repo/a.md", SourceRoot: "/repo", Text: "Alpha\n", StartLine: 1, EndLine: 1},
			{Path: "/repo/b.md", SourceRoot: "/repo", Text: "Beta\nGamma\n"},
		},
	})
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if answer.Text != "Beta is in the attached document. [source 2]" || len(answer.Sources) != 2 || !answer.CitationValid {
		t.Fatalf("answer = %+v", answer)
	}
	if answer.Sources[1].Path != "/repo/b.md" || answer.Sources[1].StartLine != 1 || answer.Sources[1].EndLine != 2 {
		t.Fatalf("source = %+v", answer.Sources[1])
	}
	want := []EventType{
		EventRunStarted,
		EventLLMRequest,
		EventLLMResponse,
		EventRunCompleted,
	}
	if got := eventTypes(events); !reflect.DeepEqual(got, want) {
		t.Fatalf("events = %v, want %v", got, want)
	}
}

func TestAnswerFromRAGSeparatesAvailableAndCitedSources(t *testing.T) {
	answer := answerFromRAG(rag.AskResult{
		Answer: "Alpha is documented. [source 2]",
		Sources: []rag.SourceChunk{
			{SourceNumber: 1, Path: "available.md", StartLine: 1, EndLine: 2},
			{SourceNumber: 2, Path: "cited.md", StartLine: 3, EndLine: 4},
		},
	})
	if len(answer.Sources) != 2 || len(answer.AvailableSources) != 2 {
		t.Fatalf("answer sources = %+v available=%+v, want all available", answer.Sources, answer.AvailableSources)
	}
	if len(answer.CitedSources) != 1 || answer.CitedSources[0].Number != 2 {
		t.Fatalf("cited sources = %+v, want source 2", answer.CitedSources)
	}
}

func TestAnswerFromAgentSeparatesAvailableAndCitedSources(t *testing.T) {
	answer := answerFromAgent(agentask.Result{
		Answer: "Agent answer. [source 2]",
		Sources: []rag.SourceChunk{
			{SourceNumber: 1, Path: "available.go", StartLine: 1, EndLine: 2},
			{SourceNumber: 2, Path: "cited.go", StartLine: 3, EndLine: 4},
		},
		ToolCalls:          []string{"search_index"},
		ThinSourceWarnings: []string{"thin-source nudge: expanding [source 2]"},
		ThinSourceNudged:   true,
	})
	if len(answer.Sources) != 2 || len(answer.AvailableSources) != 2 {
		t.Fatalf("answer sources = %+v available=%+v, want all available", answer.Sources, answer.AvailableSources)
	}
	if len(answer.CitedSources) != 1 || answer.CitedSources[0].Number != 2 {
		t.Fatalf("cited sources = %+v, want source 2", answer.CitedSources)
	}
	if len(answer.ToolCalls) != 1 || answer.ToolCalls[0].Name != "search_index" {
		t.Fatalf("tool calls = %+v", answer.ToolCalls)
	}
	if !answer.ThinSourceNudged || len(answer.ThinSourceWarnings) != 1 {
		t.Fatalf("thin-source status = %t warnings=%+v", answer.ThinSourceNudged, answer.ThinSourceWarnings)
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

func TestEngineCompactRejectsInvalidEmbeddingConcurrency(t *testing.T) {
	engine, err := NewEngine(EngineConfig{StateDir: t.TempDir()})
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	_, err = engine.Compact(context.Background(), CompactRequest{DryRun: true, EmbeddingConcurrency: -1})
	if err == nil || !strings.Contains(err.Error(), "embedding concurrency must be >= 0") {
		t.Fatalf("Compact error = %v, want embedding concurrency validation", err)
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
		SourceRoot:  "/repo",
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
