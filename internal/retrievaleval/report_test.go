package retrievaleval

import (
	"bytes"
	"errors"
	"testing"
)

type errorWriter struct{}

func (ew *errorWriter) Write(p []byte) (n int, err error) {
	return 0, errors.New("mock write error")
}

func TestWriteReportHeader(t *testing.T) {
	t.Run("Success", func(t *testing.T) {
		var buf bytes.Buffer
		name := "Test Report"
		err := writeReportHeader(&buf, name)

		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}

		expected := "\n# Test Report\n\n"
		if buf.String() != expected {
			t.Errorf("expected %q, got %q", expected, buf.String())
		}
	})

	t.Run("Error", func(t *testing.T) {
		ew := &errorWriter{}
		name := "Test Report"
		err := writeReportHeader(ew, name)

		if err == nil {
			t.Fatal("expected error, got nil")
		}

		expectedErr := "mock write error"
		if err.Error() != expectedErr {
			t.Errorf("expected error message %q, got %q", expectedErr, err.Error())
		}
	})
}
