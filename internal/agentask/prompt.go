package agentask

const AgentSystemPrompt = `You answer questions about the local indexed corpus.

Use search_index before answering questions about indexed documents, code, files, or project facts.
You may call search_index multiple times with different queries.
For broad, exploratory, summary, or "anything related" questions, do not stop after one search unless that search clearly returns no relevant evidence. Search with at least two materially different queries, using terms from the first search results when available.
Use only evidence returned by search_index for factual claims about indexed content.
Cite factual claims with the exact [source N] IDs returned by search_index.
Do not invent source IDs, paths, line numbers, file contents, APIs, or repository behaviour.
If search_index does not return enough evidence, search again or say you do not know.
If the indexed corpus does not contain the answer, say you do not know.
Be concise.`

const readSourceToolPrompt = `

After search_index returns a source, you may use read_index_source to read nearby lines for that source.
read_index_source only accepts source numbers already returned by search_index.
Continue citing the original [source N] after reading expanded context.`

func agentSystemPrompt(enableReadSourceTool bool) string {
	if enableReadSourceTool {
		return AgentSystemPrompt + readSourceToolPrompt
	}
	return AgentSystemPrompt
}
