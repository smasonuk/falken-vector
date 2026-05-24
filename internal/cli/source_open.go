package cli

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/smasonuk/falken-vector/internal/rag"
)

type SourceOpener interface {
	Open(path string, line int) error
}

type editorOpener struct {
	editor string
	run    func(name string, args ...string) error
}

var newSourceOpener = func(env func(string) string) SourceOpener {
	editor := firstEnv(env, "FALKENGO_EDITOR", "EDITOR", "VISUAL")
	if editor == "" {
		return nil
	}
	return editorOpener{
		editor: editor,
		run: func(name string, args ...string) error {
			cmd := exec.Command(name, args...)
			cmd.Stdin = os.Stdin
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
			return cmd.Run()
		},
	}
}

func (o editorOpener) Open(path string, line int) error {
	if strings.TrimSpace(o.editor) == "" {
		return fmt.Errorf("editor is not configured")
	}
	name := o.editor
	switch filepath.Base(name) {
	case "vim", "nvim", "vi":
		return o.run(name, fmt.Sprintf("+%d", line), path)
	case "code", "cursor":
		return o.run(name, "-g", fmt.Sprintf("%s:%d", path, line))
	case "zed":
		return o.run(name, fmt.Sprintf("%s:%d", path, line))
	default:
		return o.run(name, path)
	}
}

func openRetrievedSource(errOut io.Writer, chunks []rag.RetrievedChunk, sourceNumber int) error {
	sources := make([]rag.SourceChunk, 0, len(chunks))
	for i, chunk := range chunks {
		sources = append(sources, rag.SourceChunk{
			SourceNumber: i + 1,
			Path:         chunk.Path,
			StartLine:    chunk.Chunk.StartLine,
			EndLine:      chunk.Chunk.EndLine,
		})
	}
	return openAnswerSource(errOut, sources, sourceNumber)
}

func openAnswerSource(errOut io.Writer, sources []rag.SourceChunk, sourceNumber int) error {
	if sourceNumber <= 0 {
		return fmt.Errorf("--open-source must be >= 1")
	}
	for _, source := range sources {
		if source.SourceNumber == sourceNumber {
			return openSourceReference(errOut, source)
		}
	}
	return fmt.Errorf("--open-source %d out of range; available %d sources", sourceNumber, len(sources))
}

func openSourceReference(errOut io.Writer, source rag.SourceChunk) error {
	line := source.StartLine
	if line <= 0 {
		line = 1
	}
	opener := newSourceOpener(os.Getenv)
	if opener == nil {
		fmt.Fprintln(errOut, "No editor configured. Set FALKENGO_EDITOR or EDITOR.")
		fmt.Fprintln(errOut, SourceReference(source))
		return nil
	}
	if err := opener.Open(source.Path, line); err != nil {
		return err
	}
	return nil
}

func firstEnv(env func(string) string, keys ...string) string {
	for _, key := range keys {
		value := strings.TrimSpace(env(key))
		if value != "" {
			return value
		}
	}
	return ""
}
