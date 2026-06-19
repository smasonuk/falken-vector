1. **Analyze:** Understand the use of the magic number 100 in `internal/agentask/read_source_tool.go`. It represents the maximum number of context lines.
2. **Define Constants:** Add named constants `defaultContextLines = 20` and `maxContextLines = 100` near the top of the file.
3. **Refactor `readIndexSourceDescriptor`:** Update the JSON string in `readIndexSourceDescriptor()` to use `fmt.Sprintf` with these constants so the description dynamically reflects them instead of hardcoding `20` and `100`.
4. **Refactor `normalizeReadContextLines`:** Update `normalizeReadContextLines()` to return `defaultContextLines`, check `> maxContextLines`, and use `fmt.Sprintf` for the cap warning message.
5. **Verify:** Run pre-commit checks and tests to verify everything passes and functionality is unchanged.
6. **Submit:** Submit a PR with the requested format.
