package openaihttp

import "net/http"

type Doer interface {
	Do(*http.Request) (*http.Response, error)
}

type SuccessStatusClient struct {
	Next Doer
}

func (c SuccessStatusClient) Do(req *http.Request) (*http.Response, error) {
	next := c.Next
	if next == nil {
		next = http.DefaultClient
	}
	resp, err := next.Do(req)
	if err != nil || resp == nil {
		return resp, err
	}
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices && resp.StatusCode != http.StatusOK {
		resp.StatusCode = http.StatusOK
		resp.Status = "200 OK"
	}
	return resp, nil
}

type StripPlaceholderAuthorizationClient struct {
	Next             Doer
	PlaceholderToken string
}

func (c StripPlaceholderAuthorizationClient) Do(req *http.Request) (*http.Response, error) {
	next := c.Next
	if next == nil {
		next = http.DefaultClient
	}
	clone := req.Clone(req.Context())
	clone.Header = req.Header.Clone()
	if clone.Header.Get("Authorization") == "Bearer "+c.PlaceholderToken {
		clone.Header.Del("Authorization")
	}
	return next.Do(clone)
}
