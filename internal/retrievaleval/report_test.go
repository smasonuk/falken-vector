package retrievaleval

import (
	"bytes"
	"errors"
	"testing"
)

type errorWriter struct{}

func (w *errorWriter) Write(p []byte) (n int, err error) {
	return 0, errors.New("write error")
}

func TestWriteCountMetricWritesExpectedFormat(t *testing.T) {
	var buf bytes.Buffer
	err := writeCountMetric(&buf, "test-metric", 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := "test-metric: 42\n"
	if buf.String() != expected {
		t.Errorf("expected %q, got %q", expected, buf.String())
	}
}

func TestWriteCountMetricReturnsWriterError(t *testing.T) {
	err := writeCountMetric(&errorWriter{}, "test-metric", 42)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if err.Error() != "write error" {
		t.Errorf("expected 'write error', got %q", err.Error())
	}
}
