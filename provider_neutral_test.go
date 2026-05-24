package falkenvector_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestProductionCodeHasNoHardCodedProviderDefaults(t *testing.T) {
	forbidden := []string{
		"Default" + "Port" + "key" + "BaseURL",
		"Default" + "Port" + "key" + "Provider",
		"Port" + "key" + "Config",
		"X-" + "Port" + "key" + "-Provider",
		"port" + "key.syngenta.com",
		"@openai" + "-aifoundry-swc-001",
	}

	err := filepath.WalkDir(".", func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			switch entry.Name() {
			case ".git", "bin":
				return filepath.SkipDir
			default:
				return nil
			}
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		body, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, value := range forbidden {
			if strings.Contains(string(body), value) {
				t.Errorf("%s contains forbidden provider default %q", path, value)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk production code: %v", err)
	}
}
