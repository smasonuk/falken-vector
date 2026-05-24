package cli

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/smasonuk/falken-vector/internal/manifest"
	"github.com/smasonuk/falken-vector/internal/rag"
)

func TestOpenSourceRejectsOutOfRangeSourceNumber(t *testing.T) {
	err := openRetrievedSource(&bytes.Buffer{}, []rag.RetrievedChunk{sourceOpenChunk()}, 2)
	if err == nil || !strings.Contains(err.Error(), "out of range") {
		t.Fatalf("openRetrievedSource error = %v, want out of range", err)
	}
}

func TestOpenSourceUsesConfiguredEditor(t *testing.T) {
	old := newSourceOpener
	defer func() { newSourceOpener = old }()
	opener := &recordingSourceOpener{}
	newSourceOpener = func(func(string) string) SourceOpener {
		return opener
	}
	if err := openRetrievedSource(&bytes.Buffer{}, []rag.RetrievedChunk{sourceOpenChunk()}, 1); err != nil {
		t.Fatalf("openRetrievedSource: %v", err)
	}
	if opener.path != "internal/rag/retrieve.go" || opener.line != 35 {
		t.Fatalf("opener = %+v, want path and start line", opener)
	}
}

func TestOpenSourceNoEditorPrintsReference(t *testing.T) {
	old := newSourceOpener
	defer func() { newSourceOpener = old }()
	newSourceOpener = func(func(string) string) SourceOpener { return nil }
	var errOut bytes.Buffer
	if err := openRetrievedSource(&errOut, []rag.RetrievedChunk{sourceOpenChunk()}, 1); err != nil {
		t.Fatalf("openRetrievedSource: %v", err)
	}
	output := errOut.String()
	if !strings.Contains(output, "No editor configured") || !strings.Contains(output, "[source 1] internal/rag/retrieve.go:35-73") {
		t.Fatalf("output = %q, want warning and source reference", output)
	}
}

func TestOpenSourceFormatsCodeEditorCommand(t *testing.T) {
	var name string
	var args []string
	opener := editorOpener{
		editor: "code",
		run: func(gotName string, gotArgs ...string) error {
			name = gotName
			args = append([]string(nil), gotArgs...)
			return nil
		},
	}
	if err := opener.Open("internal/rag/retrieve.go", 35); err != nil {
		t.Fatalf("Open: %v", err)
	}
	if name != "code" || len(args) != 2 || args[0] != "-g" || args[1] != "internal/rag/retrieve.go:35" {
		t.Fatalf("command = %s %+v, want code -g path:line", name, args)
	}
}

func TestOpenSourceUsesFalkengoEditorBeforeEditorFallback(t *testing.T) {
	editor := firstEnv(func(key string) string {
		switch key {
		case "FALKENGO_EDITOR":
			return "cursor"
		case "EDITOR":
			return "vim"
		default:
			return ""
		}
	}, "FALKENGO_EDITOR", "EDITOR", "VISUAL")
	if editor != "cursor" {
		t.Fatalf("editor = %q, want FALKENGO_EDITOR first", editor)
	}
}

func TestOpenSourceUsesEditorFallback(t *testing.T) {
	editor := firstEnv(func(key string) string {
		if key == "EDITOR" {
			return "vim"
		}
		return ""
	}, "FALKENGO_EDITOR", "EDITOR", "VISUAL")
	if editor != "vim" {
		t.Fatalf("editor = %q, want EDITOR fallback", editor)
	}
}

func TestOpenSourceReturnsEditorError(t *testing.T) {
	old := newSourceOpener
	defer func() { newSourceOpener = old }()
	newSourceOpener = func(func(string) string) SourceOpener {
		return failingSourceOpener{}
	}
	err := openRetrievedSource(&bytes.Buffer{}, []rag.RetrievedChunk{sourceOpenChunk()}, 1)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("error = %v, want editor error", err)
	}
}

func sourceOpenChunk() rag.RetrievedChunk {
	return rag.RetrievedChunk{
		Path: "internal/rag/retrieve.go",
		Chunk: manifest.Chunk{
			StartLine: 35,
			EndLine:   73,
		},
	}
}

type recordingSourceOpener struct {
	path string
	line int
}

func (o *recordingSourceOpener) Open(path string, line int) error {
	o.path = path
	o.line = line
	return nil
}

type failingSourceOpener struct{}

func (failingSourceOpener) Open(string, int) error {
	return errors.New("boom")
}
