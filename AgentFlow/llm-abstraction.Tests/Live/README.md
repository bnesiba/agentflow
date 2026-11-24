# Live provider tests

Run these xUnit tests from the repository root to exercise the real OpenAI Responses, Anthropic Messages, Gemini Interactions, and Gemini generateContent endpoints. Normal test runs skip the live suite. Selecting a provider requires its API key and an explicit opt-in.

## Run

For one-time setup, copy `live.example.runsettings` beside the test project to `live.local.runsettings` if the local file does not already exist. Enter the keys for the providers you want to test between the matching XML tags. Both Gemini surfaces use `GEMINI_API_KEY`. Git ignores the local file.

Passing the local settings file opts that run into billed calls. From the repository root, start with one feature and provider.

```powershell
$settings = 'AgentFlow/llm-abstraction.Tests/live.local.runsettings'
dotnet test AgentFlow/llm-abstraction.Tests --settings $settings --filter 'Category=Live&DisplayName~Instructions_counting&DisplayName~OpenAI'
```

The first example makes one input-token count and one short generation. Use these commands to inspect the cases, run one provider, or run the complete suite.

```powershell
dotnet test AgentFlow/llm-abstraction.Tests --settings $settings --list-tests --filter 'Category=Live'
dotnet test AgentFlow/llm-abstraction.Tests --settings $settings --filter 'Category=Live&DisplayName~Claude'
dotnet test AgentFlow/llm-abstraction.Tests --settings $settings --filter 'Category=Live&DisplayName~Gemini'
dotnet test AgentFlow/llm-abstraction.Tests --settings $settings --filter 'Category=Live'
```

## Coverage

| Feature | Live check |
| --- | --- |
| Generation and counting | System and user turns, instructions, output limits, metadata on supported providers, native token counts and accuracy labels, usage, admission, and transport status |
| Streaming and cancellation | Text deltas agree with the assembled message, terminal status and usage are delivered, and cancelling an open stream raises caller cancellation |
| Structured output | Parsed JSON satisfies a closed object schema on every surface. OpenAI and Gemini also get a plain JSON-object case |
| Stopping and errors | Stop sequences truncate output on Claude and Gemini. A nonexistent model exercises each provider's HTTP error normalization |
| Tools and history | Required, specific, auto, and none choices, strict and parallel controls where supported, streamed arguments, native assistant replay, successful and failed tool results, and two-turn stored and stateless continuation |
| Search and evidence | Actual hosted web and image searches, source attribution, provider filters and limits, and Claude citations from supplied documents and search results |
| Media | Image and PDF contents, file-ID or URI references, and inline Gemini audio and video. Uploaded PDFs are deleted in a finally block |
| Reasoning | Effort and summary requests on both Gemini surfaces, adaptive and manual Claude thinking, and OpenAI pro mode and reasoning context |
| Caching | Repeated eligible prefixes with reported cache reads on OpenAI and Claude. Gemini explicit cache creation, retrieval, listing, expiration update, generation against the cached prefix, and deletion |

The common models are `gpt-6-luna`, `claude-haiku-5-5`, and `gemini-3.5-flash-lite`. Individual cases use `gpt-5.6-luna` for persisted reasoning context and `claude-sonnet-4-6` for manual thinking. Both Gemini image-search cases use `gemini-nano-banana-2.1`. Most requests cap output at 128 tokens. Search, reasoning, and cache cases use larger budgets where the feature requires them. Stored-turn tests opt into provider storage. The Gemini cache also has a short expiration in case the test process stops during cleanup.

A selected case fails on missing credentials, missing model or tool access, or a provider response that breaks an assertion. Search selection, cache hits, and generated answers can vary across runs. The offline contract tests cover precise wire fields, unsupported combinations, malformed streams, rate limits, and transport failures that a live test cannot reliably induce.
