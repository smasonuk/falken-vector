package retrievaleval

import (
	"strings"
	"testing"
)

func TestLoadJSONLLoadsValidDataset(t *testing.T) {
	input := `
{"id":"one","question":"Where is retrieval?","expected_chunk_ids":["chunk-1"],"notes":"human note"}

{"id":"two","question":"Where is the CLI?","expected_path_suffixes":["internal/cli/query.go"]}
`
	cases, err := LoadJSONL(strings.NewReader(input))
	if err != nil {
		t.Fatalf("LoadJSONL: %v", err)
	}
	if len(cases) != 2 {
		t.Fatalf("cases = %d, want 2", len(cases))
	}
	if cases[0].ID != "one" || cases[0].ExpectedChunkIDs[0] != "chunk-1" {
		t.Fatalf("case[0] = %+v", cases[0])
	}
}

func TestLoadJSONLRejectsInvalidJSONWithLineNumber(t *testing.T) {
	_, err := LoadJSONL(strings.NewReader("\n{\"question\":\n"))
	if err == nil || !strings.Contains(err.Error(), "line 2: invalid JSON") {
		t.Fatalf("LoadJSONL error = %v, want line-numbered JSON error", err)
	}
}

func TestLoadJSONLRejectsMissingQuestion(t *testing.T) {
	_, err := LoadJSONL(strings.NewReader(`{"id":"bad","expected_chunk_ids":["chunk"]}`))
	if err == nil || !strings.Contains(err.Error(), "line 1: question is required") {
		t.Fatalf("LoadJSONL error = %v, want missing question", err)
	}
}

func TestLoadJSONLRejectsMissingExpectations(t *testing.T) {
	_, err := LoadJSONL(strings.NewReader(`{"id":"bad","question":"hello"}`))
	if err == nil || !strings.Contains(err.Error(), "line 1: at least one") {
		t.Fatalf("LoadJSONL error = %v, want missing expectations", err)
	}
}

func TestLoadJSONLTrimsBlankExpectationEntries(t *testing.T) {
	_, err := LoadJSONL(strings.NewReader(`{"question":"hello","expected_chunk_ids":[" "]}`))
	if err == nil || !strings.Contains(err.Error(), "line 1: at least one") {
		t.Fatalf("LoadJSONL error = %v, want missing expectations after trimming", err)
	}
}

func TestLoadJSONLRejectsEmptyDataset(t *testing.T) {
	_, err := LoadJSONL(strings.NewReader("\n\n"))
	if err == nil || !strings.Contains(err.Error(), "dataset contains no cases") {
		t.Fatalf("LoadJSONL error = %v, want empty dataset error", err)
	}
}
