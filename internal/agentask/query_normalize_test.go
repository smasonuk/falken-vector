package agentask

import (
	"testing"
)

func TestNormalizeSearchQuery(t *testing.T) {
	tests := []struct {
		name  string
		query string
		want  string
	}{
		{
			name:  "empty string",
			query: "",
			want:  "",
		},
		{
			name:  "only whitespace",
			query: "   \t \n ",
			want:  "",
		},
		{
			name:  "only punctuation",
			query: ".,:;()[]{}",
			want:  "",
		},
		{
			name:  "basic query",
			query: "hello world",
			want:  "hello world",
		},
		{
			name:  "mixed casing and duplicates",
			query: "Hello hello world WORLD",
			want:  "Hello world",
		},
		{
			name:  "punctuation stripping",
			query: "[hello], (world).",
			want:  "hello world",
		},
		{
			name:  "middle punctuation",
			query: "hello,world",
			want:  "hello,world",
		},
		{
			name:  "newlines and tabs",
			query: "hello\n\tworld",
			want:  "hello world",
		},
		{
			name:  "preserves first seen casing",
			query: "woRLD world World",
			want:  "woRLD",
		},
		{
			name:  "mixed duplicates with punctuation",
			query: "test [test] (test) test. test,",
			want:  "test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeSearchQuery(tt.query); got != tt.want {
				t.Errorf("normalizeSearchQuery(%q) = %q, want %q", tt.query, got, tt.want)
			}
		})
	}
}
