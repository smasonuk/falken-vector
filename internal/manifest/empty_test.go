package manifest

import (
	"context"
	"testing"
)

func TestEmptyStoreGetDocumentByPath(t *testing.T) {
	store := EmptyStore{}
	doc, err := store.GetDocumentByPath(context.Background(), "some/path")
	if err != ErrNotFound {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
	if doc != nil {
		t.Errorf("expected doc to be nil, got %v", doc)
	}
}
