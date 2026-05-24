package retrievaleval

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

type Case struct {
	ID                   string   `json:"id"`
	Question             string   `json:"question"`
	ExpectedChunkIDs     []string `json:"expected_chunk_ids"`
	ExpectedPaths        []string `json:"expected_paths"`
	ExpectedPathSuffixes []string `json:"expected_path_suffixes"`
	Notes                string   `json:"notes,omitempty"`
}

func LoadJSONL(r io.Reader) ([]Case, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var cases []Case
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var c Case
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			return nil, fmt.Errorf("line %d: invalid JSON: %w", lineNo, err)
		}
		normalizeCase(&c)
		if strings.TrimSpace(c.Question) == "" {
			return nil, fmt.Errorf("line %d: question is required", lineNo)
		}
		if len(c.ExpectedChunkIDs) == 0 && len(c.ExpectedPaths) == 0 && len(c.ExpectedPathSuffixes) == 0 {
			return nil, fmt.Errorf("line %d: at least one expected_chunk_ids, expected_paths, or expected_path_suffixes entry is required", lineNo)
		}
		cases = append(cases, c)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read dataset: %w", err)
	}
	if len(cases) == 0 {
		return nil, fmt.Errorf("dataset contains no cases")
	}
	return cases, nil
}

func normalizeCase(c *Case) {
	c.ID = strings.TrimSpace(c.ID)
	c.Question = strings.TrimSpace(c.Question)
	c.ExpectedChunkIDs = compactStrings(c.ExpectedChunkIDs)
	c.ExpectedPaths = compactStrings(c.ExpectedPaths)
	c.ExpectedPathSuffixes = compactStrings(c.ExpectedPathSuffixes)
}

func compactStrings(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}
