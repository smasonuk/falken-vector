package openaihttp

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type doerFunc func(*http.Request) (*http.Response, error)

func (f doerFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestSuccessStatusClientNormalizesNonstandard2xx(t *testing.T) {
	client := SuccessStatusClient{Next: doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: 246,
			Status:     "246 Custom Success",
			Body:       io.NopCloser(strings.NewReader(`{"ok":true}`)),
		}, nil
	})}

	resp, err := client.Do(httptest.NewRequest(http.MethodPost, "/chat/completions", nil))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if resp.StatusCode != http.StatusOK || resp.Status != "200 OK" {
		t.Fatalf("status = %d %q, want 200 OK", resp.StatusCode, resp.Status)
	}
}

func TestSuccessStatusClientPreservesErrorStatus(t *testing.T) {
	client := SuccessStatusClient{Next: doerFunc(func(*http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusBadRequest,
			Status:     "400 Bad Request",
			Body:       io.NopCloser(strings.NewReader(`{"error":"bad"}`)),
		}, nil
	})}

	resp, err := client.Do(httptest.NewRequest(http.MethodPost, "/chat/completions", nil))
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if resp.StatusCode != http.StatusBadRequest || resp.Status != "400 Bad Request" {
		t.Fatalf("status = %d %q, want 400 Bad Request", resp.StatusCode, resp.Status)
	}
}

func TestStripPlaceholderAuthorizationClient(t *testing.T) {
	var got string
	client := StripPlaceholderAuthorizationClient{
		PlaceholderToken: "placeholder-token",
		Next: doerFunc(func(req *http.Request) (*http.Response, error) {
			got = req.Header.Get("Authorization")
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Body:       io.NopCloser(strings.NewReader(`{}`)),
			}, nil
		}),
	}
	req := httptest.NewRequest(http.MethodPost, "/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer placeholder-token")

	_, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if got != "" {
		t.Fatalf("Authorization = %q, want stripped", got)
	}
	if original := req.Header.Get("Authorization"); original != "Bearer placeholder-token" {
		t.Fatalf("original Authorization = %q, want request unchanged", original)
	}
}

func TestStripPlaceholderAuthorizationClientPreservesCallerAuthorization(t *testing.T) {
	var got string
	client := StripPlaceholderAuthorizationClient{
		PlaceholderToken: "placeholder-token",
		Next: doerFunc(func(req *http.Request) (*http.Response, error) {
			got = req.Header.Get("Authorization")
			return &http.Response{
				StatusCode: http.StatusOK,
				Status:     "200 OK",
				Body:       io.NopCloser(strings.NewReader(`{}`)),
			}, nil
		}),
	}
	req := httptest.NewRequest(http.MethodPost, "/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer caller-token")

	_, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if got != "Bearer caller-token" {
		t.Fatalf("Authorization = %q, want caller token preserved", got)
	}
}
