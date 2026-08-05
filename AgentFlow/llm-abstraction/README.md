# LLM Abstraction Layer for C#

A .NET 8 abstraction over these HTTP APIs:

- Anthropic Claude Messages API (`POST /v1/messages`)
- OpenAI Responses API (`POST /v1/responses`)
- Google Gemini Interactions API, with a `generateContent` compatibility path

The portable subset covers text, images/files, custom function tools, provider-native Web Search, structured output, reasoning controls, and streaming. Provider-native state is retained where it is required for correct multi-turn continuation.

## Compatibility status

| Capability | Claude | OpenAI | Gemini |
|---|---|---|---|
| Text generation | Supported | Supported | Supported |
| Text streaming | Supported | Supported | Supported |
| Custom function calls | Supported | Supported | Supported |
| Streaming function calls | Supported | Supported | Supported |
| Parallel function calls | Preserved | Configurable | Preserved |
| Images | URL/base64 | URL/base64/file ID | Base64/Files API URI |
| Documents/files | URL/base64/file ID | URL/base64/file ID | Base64/Files API URI |
| JSON object mode | Not exposed; use a schema | Supported | Supported |
| JSON Schema output | Provider schema subset | Provider schema subset | Provider schema subset |
| Reasoning controls | Manual/adaptive subset | Effort/summary subset | Budget/level subset |
| Native reasoning continuation | Preserved | Preserved | Preserved |
| Provider-native Web Search | Supported | Supported | Supported |
| Search calls/results | Portable + native | Portable + native | Portable + native |
| Citations/grounding | Portable + native | Portable + native | Portable + native |
| Preflight input counting | Estimated | Exact | Exact on `generateContent`; estimated for Interactions |
| Retries/rate-limit metadata | Supported | Supported | Supported; raw limits when undocumented |

Model capability data is advisory. Documented incompatibilities and unknown models produce warnings but do not block serialization, so new model names do not require a package update. Hard errors are reserved for API-shape settings that cannot be represented without dropping or changing the request.

Gemini uses the Interactions API by default. Sampling controls, a legacy thinking budget, or a Web Search time-range filter automatically use `generateContent` because those fields are not exposed by Interactions.

## Basic usage

```csharp
using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;

var service = LLMServiceFactory.CreateOpenAI("your-api-key");

var request = new UnifiedRequest
{
    Model = "gpt-5.6",
    Instructions = "Answer clearly and concisely.",
    Messages =
    {
        new UnifiedMessage(MessageRole.User, "What is the capital of France?")
    },
    Parameters = new GenerationParameters
    {
        MaxOutputTokens = 200,
        Temperature = 0.4
    }
};

var response = await service.GenerateAsync(request);
var text = response.Choices[0].Message.Content
    .OfType<TextContent>()
    .Select(content => content.Text);

Console.WriteLine(string.Concat(text));
```

Model IDs change over time and may not be enabled for every account. Verify the selected ID in the provider's current model documentation.

## Providers

```csharp
var openAI = LLMServiceFactory.CreateOpenAI("openai-key");

var claude = LLMServiceFactory.CreateClaude(
    apiKey: "anthropic-key",
    apiVersion: "2023-06-01");

var gemini = LLMServiceFactory.CreateGemini("gemini-key");
```

Custom base URLs are supported by each factory method. Gemini authentication uses the `x-goog-api-key` header; the key is not placed in the URL.

## Instructions and messages

Use `UnifiedRequest.Instructions` for developer/system-level instructions. Text from any `MessageRole.System` messages is appended to `Instructions` in request order, separated by blank lines, and removed from conversational history before provider conversion.

Non-text system content cannot be represented portably and is rejected rather than silently discarded.

Portable roles are:

- `System`
- `User`
- `Assistant`
- `Tool`

Portable content includes:

- `TextContent`
- `RefusalContent`
- `ImageContent`
- `MediaContent`
- `ToolCallContent`
- `ToolResultContent`
- `ProviderToolCallContent`
- `ProviderToolResultContent`
- `ProviderNativeContent`

`ProviderNativeContent` and `NativeRepresentation` carry opaque signed or future provider blocks. Native content is provider-bound and cannot be sent to a different provider.

## Custom tools

```csharp
var weather = new ToolDefinition
{
    Name = "get_weather",
    Description = "Get the current weather for a city.",
    Strict = true,
    Parameters = new Dictionary<string, object>
    {
        ["type"] = "object",
        ["properties"] = new Dictionary<string, object>
        {
            ["city"] = new Dictionary<string, object>
            {
                ["type"] = "string"
            }
        },
        ["required"] = new[] { "city" },
        ["additionalProperties"] = false
    }
};

request.Tools = new List<ToolDefinition> { weather };
request.ToolChoice = new ToolChoice { Type = ToolChoiceType.Auto };
```

Tool choice types are `Auto`, `None`, `Required`, and `Specific`. `Specific` requires a valid `ToolName`.

Claude thinking cannot be combined with forced (`Required` or `Specific`) tool choice. Validation rejects this combination before the HTTP call.

## Provider-native Web Search

Function tools and provider-native tools share one collection. Your application executes a `FunctionTool`; the currently implemented provider-native capability, Web Search, is executed by Anthropic, OpenAI, or Gemini. Future native capabilities such as Computer Use or local shell can still require an application-side execution loop because provider-native describes the API protocol, not execution ownership.

```csharp
request.Tools = new ToolCollection
{
    weather,
    ProviderTools.WebSearch(new WebSearchOptions
    {
        AllowedDomains = new[] { "example.com" },
        Location = new ApproximateLocation
        {
            Country = "US",
            City = "Boston",
            Region = "Massachusetts"
        }
    })
};
```

The same `ProviderTools.WebSearch(...)` declaration maps to Anthropic `web_search`, OpenAI Responses `web_search`, or Gemini `google_search`. Portable controls are enforced only when the selected provider supports them:

| Control | Claude | OpenAI | Gemini |
|---|---|---|---|
| Allowed domains | Supported | Supported | Rejected as unsupported |
| Blocked domains | Supported | Supported | Rejected as unsupported |
| Approximate location | Supported | Supported | Rejected as unsupported |
| Web results | Supported | Supported | Supported |
| Image results | Rejected as unsupported | Supported | Supported |

Provider-specific controls remain typed and isolated under `WebSearchOptions.Anthropic`, `.OpenAI`, and `.Gemini`. These include Anthropic maximum uses/dynamic filtering/full-result inclusion, OpenAI context size/source inclusion/live-web access/token budget/image settings, and Gemini start/end time. Gemini time ranges require both endpoints, must be ordered, and trigger the `generateContent` compatibility path automatically.

Provider-managed search activity appears as `ProviderToolCallContent` and `ProviderToolResultContent`; it never appears as `ToolCallContent`, so callers do not accidentally execute it. Results expose portable `WebSource` objects. `UnifiedMessage.Evidence` provides stable sources, typed locations, grouped grounding supports, and required attribution artifacts; `TextContent.Citations` remains the convenient per-block view. Every projection also retains its native provider JSON.

See [PROVIDER_NATIVE_TOOLS.md](PROVIDER_NATIVE_TOOLS.md) for the complete architecture notes, provider tool inventory, current limitations, and checklist for adding the next native capability.

See [MODEL_CAPABILITIES.md](MODEL_CAPABILITIES.md) for model/endpoint capability resolution, advisory warning behavior, reasoning mappings, and the portable JSON Schema contract.

See [REMAINING_WORK_PLAN.md](REMAINING_WORK_PLAN.md) for the prioritized plan covering model-aware reasoning/structured output, retries and rate limits, token counting, citations/grounding, prompt caching, embeddings, batch, and realtime.

Applications that display grounded answers remain responsible for rendering citations and every `AttributionArtifact` whose `DisplayRequired` value is true. Anthropic may return `FinishReason.Pause` during a long server-tool turn; continue by replaying the returned assistant message unchanged with the same tool definition.

### Returning tool results

```csharp
var call = response.Choices[0].Message.Content
    .OfType<ToolCallContent>()
    .First();

var resultMessage = new UnifiedMessage(
    MessageRole.Tool,
    new List<ContentBlock>
    {
        new ToolResultContent
        {
            ToolCallId = call.Id,
            ToolName = call.Name,
            Output = new { temperature = 72 }
        }
    });
```

For Gemini, the converter can resolve an omitted `ToolName` from a matching prior `ToolCallId`. An orphaned Gemini result must supply `ToolName`. A mismatched explicit name is rejected.

## Provider-native continuation

Modern provider responses contain state that cannot safely be flattened into text and function calls:

- Claude signed `thinking` and `redacted_thinking` blocks
- OpenAI reasoning and other Responses output items
- Gemini ordered parts and `thoughtSignature` values

The abstraction preserves this state without translating it between providers.

### Claude and Gemini

Append the returned assistant message, including its native representations, followed by the tool result:

```csharp
nextRequest.Messages.Add(response.Choices[0].Message);
nextRequest.Messages.Add(resultMessage);
```

Do not rebuild or reorder signed provider blocks.

### OpenAI stateless replay

OpenAI native output items are stored in `UnifiedResponse.Continuation`. For stateless replay, attach that state and send only new input—the retained output items already contain the previous assistant response:

```csharp
var nextRequest = new UnifiedRequest
{
    Model = request.Model,
    Continuation = response.Continuation,
    Messages = { resultMessage }
};
```

Do not also append the projected OpenAI assistant message; that would duplicate the retained output items.

### OpenAI server-managed continuation

```csharp
response.Continuation!.Mode = ContinuationMode.ServerManaged;

var nextRequest = new UnifiedRequest
{
    Model = request.Model,
    Continuation = response.Continuation,
    Messages = { resultMessage }
};
```

This emits `previous_response_id` and does not replay the native items.

## Streaming

```csharp
await foreach (var chunk in service.StreamAsync(request))
{
    if (chunk.Delta.Content != null)
        Console.Write(chunk.Delta.Content);

    foreach (var call in chunk.Delta.ToolCalls ?? Enumerable.Empty<ToolCallDelta>())
    {
        if (call.Arguments != null)
        {
            // Incremental JSON fragment. Append by call.Index/call.Id.
        }

        if (call.IsComplete)
        {
            // call.CompleteArguments is the completed JSON argument object.
            // call.Id is the provider continuation/call ID.
            // call.ItemId is a separate provider response-item ID when applicable.
        }
    }

    if (chunk.CompletedMessage != null)
    {
        // Fully accumulated assistant message, including native signed content.
    }
}
```

Streaming fields:

- `Delta.Content`: display-oriented text fragment
- `Delta.ToolCalls`: tool lifecycle deltas
- `Delta.ContentBlocks`: complete native blocks observed in the event
- `CompletedMessage`: accumulated assistant message at a terminal event
- `Continuation`: response-level native continuation state when provided
- `Usage`: provider usage when available
- `FinishReason`: normalized terminal reason

## Structured output

```csharp
request.Output = new OutputFormat
{
    Kind = OutputFormatKind.JsonSchema,
    JsonSchema = new JsonSchemaDefinition
    {
        Name = "person",
        Schema = new Dictionary<string, object>
        {
            ["type"] = "object",
            ["properties"] = new Dictionary<string, object>
            {
                ["name"] = new Dictionary<string, object> { ["type"] = "string" },
                ["age"] = new Dictionary<string, object> { ["type"] = "integer" }
            },
            ["required"] = new[] { "name", "age" },
            ["additionalProperties"] = false
        }
    }
};
```

`OutputFormatKind.JsonObject` requests JSON without a caller-supplied schema where the provider exposes that mode. Anthropic has no equivalent mode, so use `JsonSchema` there. Schema-constrained output validates a portable, lossless provider intersection before sending: object roots, closed objects, all properties required (nullable unions represent optional values), supported primitive/union types, arrays, enums, `anyOf`, definitions/references, and the common date/time formats. Constraints outside that intersection are rejected instead of silently removed.

## Reasoning controls

```csharp
request.Reasoning = new ReasoningOptions
{
    Effort = ReasoningEffort.Medium,
    Output = ReasoningOutput.Summary,
    OpenAI = new OpenAIReasoningOptions
    {
        Mode = OpenAIReasoningMode.Pro,
        Context = OpenAIReasoningContext.AllTurns,
        Summary = OpenAIReasoningSummary.Auto
    }
};
```

Portable effort and summary intent work across providers where available. Non-equivalent controls are typed extensions:

- Claude: `AnthropicReasoningOptions.Mode` selects default, disabled, adaptive, or manual thinking; manual mode carries `BudgetTokens`. Effort is serialized separately in `output_config.effort`.
- OpenAI: `OpenAIReasoningOptions` adds standard/pro mode, persisted context, and summary style.
- Gemini: portable effort maps to `thinking_level`; `GeminiReasoningOptions.ThinkingBudget` is available only on the legacy `generateContent` surface.

Gemini budget and level controls cannot be supplied together. Claude manual budgets and thinking/sampling combinations must satisfy API constraints. Model-specific mismatches produce warnings and are still sent.

## Provider options

Provider-specific root fields can be added without changing the portable model:

```csharp
request.ProviderOptions = new ProviderOptions
{
    OpenAI = new Dictionary<string, object>
    {
        ["store"] = false,
        ["service_tier"] = "flex",
        ["include"] = new[] { "reasoning.encrypted_content" }
    }
};
```

Equivalent dictionaries exist for Claude and Gemini. Values are serialized as root request fields with their exact supplied names.

Unified fields such as `model`, `input`, `messages`, `generationConfig`, and `tools` cannot be overridden through provider options. Collisions throw an exception. Claude `anthropicBeta` is handled as an HTTP header, and Gemini `safetySettings` is handled by its typed request property.

Provider options bypass most semantic capability checks. Confirm native fields against current provider documentation.

## Request validation

Converters validate requests before serialization. Validation includes:

- Required model and conversation input
- Sampling ranges
- Token ranges
- Tool names, duplicate tools, and tool choice
- Tool-result call IDs
- Structured-output schema presence
- Portable JSON Schema vocabulary and strict-shape requirements
- Model/endpoint capability warnings
- Provider-specific reasoning/tool conflicts
- Provider continuation mismatches

Inspect diagnostics without sending a request:

```csharp
using LLMAbstraction.Core.Validation;

var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.OpenAI);
var capabilities = LLMRequestValidator.GetCapabilities(
    LLMProvider.OpenAI,
    request.Model,
    LLMApiSurface.OpenAIResponses);
```

Converters throw `LLMRequestValidationException` when error diagnostics exist. Model capability warnings never block conversion and remain available through explicit validation.

Capability profiles report the resolved family, recognition state, API surface, structured-output/reasoning support, accepted reasoning efforts, and relevant endpoint controls. See [MODEL_CAPABILITIES.md](MODEL_CAPABILITIES.md) for the support matrix and update policy.

## Transport reliability and rate limits

Generation, streaming, and token-count calls share one transport implementation. By default it retries up to three attempts for request timeouts, connection failures, HTTP 408/409/429, and 5xx responses. It honors provider `Retry-After` or reset hints, applies bounded exponential backoff with jitter, recreates each HTTP request, and never retries validation, authentication, or permission failures.

```csharp
using LLMAbstraction.Core.Transport;

var service = LLMServiceFactory.CreateOpenAI(
    "openai-key",
    transport: new LLMTransportOptions
    {
        Retry = new RetryPolicy
        {
            MaximumAttempts = 4,
            MaximumElapsedTime = TimeSpan.FromSeconds(45),
            BaseDelay = TimeSpan.FromMilliseconds(250),
            MaximumDelay = TimeSpan.FromSeconds(10)
        },
        AdmissionPolicy = myAdmissionPolicy,
        RetryObserver = myRetryObserver
    });
```

`IRequestAdmissionPolicy` is an optional application-owned hook for local concurrency limits, distributed throttling, or budget admission. Its lease covers the whole logical request, including retries and streaming enumeration. The library does not impose a process-local limiter that would be misleading in distributed applications.

One request may override retry behavior and provide a token estimate to admission control without changing the provider payload:

```csharp
request.Transport = new RequestTransportOptions
{
    Retry = new RetryPolicy { Enabled = false },
    EstimatedInputTokens = 12_345
};
```

`UnifiedResponse.Transport`, the first emitted `StreamChunk.Transport`, `TokenCountResult.Transport`, and structured exceptions expose request ID, every attempt, and the final response's normalized rate-limit snapshot. OpenAI request/token windows and Anthropic request/input/output/aggregate-token windows are normalized; all rate-limit and retry headers are also retained in `RateLimitSnapshot.RawHeaders`. Gemini does not currently document standard response limit headers, so any returned limit headers are preserved raw rather than assigned invented semantics.

A stream is retried only while obtaining its initial HTTP response. Once the body is being parsed, malformed data, connection loss, or any other partial-stream failure is surfaced and the stream is never restarted, preventing duplicate visible output.

## Preflight token counting

Factory-created services implement both `ILLMService` and the separate `ITokenCountingService` contract:

```csharp
var count = await service.CountInputTokensAsync(request);

Console.WriteLine(count.InputTokens);
Console.WriteLine(count.Accuracy); // Exact or Estimate

request.Transport = new RequestTransportOptions
{
    EstimatedInputTokens = count.InputTokens
};
var response = await service.GenerateAsync(request);
```

Counting reuses the provider converters, so instructions, conversation history, tools, schemas, media, reasoning settings, and native continuation are included as the provider will see them. OpenAI uses `POST /responses/input_tokens`. Anthropic uses `POST /v1/messages/count_tokens` and is marked `Estimate` because Anthropic documents that the eventual Messages input usage can differ slightly. Gemini uses `models/{model}:countTokens` with the full converted `generateContentRequest`; its result is `Exact` when generation uses `generateContent`, but `Estimate` when generation uses Interactions because the provider exposes no Interactions-shaped counting endpoint. `ProviderMetadata` identifies both Gemini surfaces.

The operation predicts input tokens only. It does not claim to predict generated output tokens or enforce context limits for model IDs known or unknown to the package.

## Evidence, citations, and grounding

Each assistant message has an `EvidenceCollection` with three deliberately separate concepts:

- `EvidenceSource` is a stable response-local web, file, document, image, place, media, or search-result identity.
- `Citation` links an answer-text span to one or more source IDs and typed source locations.
- `GroundingSupport` represents provider grounding where one answer span is supported by multiple chunks and optional per-source confidence values.
- `AttributionArtifact` carries provider-required HTML/widget display material; it is not treated as evidence.

Source locations are typed as character, page, content-block, timestamp, or URI-fragment ranges. Each range records its own index unit and inclusive/exclusive semantics instead of giving universal meaning to ambiguous `StartIndex`/`EndIndex` fields. Provider-native JSON remains attached for fields with no portable equivalent.

Anthropic citable input documents use `DocumentContent` with optional `Title`, `Context`, and `CitationsEnabled`. `SearchResultContent` represents Anthropic's citable retrieved-result input. Other providers may still accept an ordinary `MediaContent`/`DocumentContent` file, but citation-specific fields fail validation when their API surface cannot represent them; they are never silently removed. Anthropic citation-enabled inputs also explicitly conflict with schema-constrained output.

## Prompt caching

Prompt caching has request intent, semantic breakpoints, and explicit resource lifecycle as separate layers:

```csharp
request.Cache = new PromptCacheOptions
{
    Mode = PromptCacheMode.PreferReuse,
    Ttl = PromptCacheTtl.OneHour
};
request.InstructionsCache = new PromptCacheDirective
{
    Ttl = PromptCacheTtl.OneHour
};
request.Messages[0].Content[^1].Cache = new PromptCacheDirective();
```

- Anthropic maps `PreferReuse` to top-level automatic caching and supports cache directives on tools, instructions, and message content. Validation enforces its four-breakpoint limit, supported five-minute/one-hour TTLs, and long-before-short ordering.
- OpenAI implicit caching needs no breakpoints. Newer Responses models can use explicit `prompt_cache_breakpoint` items plus typed cache key, mode, and 30-minute request TTL; legacy retention remains a typed OpenAI extension. Unsupported or unverified model combinations warn but still serialize.
- Gemini Interactions uses implicit caching only. A `cachedContent` resource reference automatically routes generation to `generateContent`, the surface that can represent it.

Unknown model IDs are never blocked by a package model allowlist. Model-specific support is advisory; malformed TTLs, impossible field placement, and endpoint features that cannot be represented remain validation errors.

Gemini explicit resources use the separate `IPromptCacheService` contract:

```csharp
var caches = LLMServiceFactory.CreateGeminiPromptCache("gemini-key");
var cache = await caches.CreatePromptCacheAsync(new PromptCacheCreateRequest
{
    Prefix = stablePrefixRequest,
    DisplayName = "product-manual",
    Ttl = TimeSpan.FromHours(1)
});

request.Cache = new PromptCacheOptions
{
    Gemini = new GeminiPromptCacheOptions { CachedContentName = cache.Name }
};
```

The service supports create, get, paginated list, expiration update, and delete. Cache usage is normalized as read/write tokens, including Anthropic five-minute/one-hour writes where returned, while native usage remains available.

## Errors

Provider HTTP failures throw `LLMApiException`, which derives from `HttpRequestException` for compatibility.

```csharp
using LLMAbstraction.Core.Errors;

try
{
    await service.GenerateAsync(request);
}
catch (LLMApiException exception)
{
    Console.WriteLine(exception.Provider);
    Console.WriteLine(exception.StatusCode);
    Console.WriteLine(exception.ErrorCode);
    Console.WriteLine(exception.ErrorType);
    Console.WriteLine(exception.Parameter);
    Console.WriteLine(exception.RequestId);
    Console.WriteLine(exception.RetryAfter);
    Console.WriteLine(exception.IsTransient);
    Console.WriteLine(exception.Transport?.Attempts.Count);
}
```

The raw response and parsed provider detail JSON remain available for diagnostics.

## Known portability limits

- Web Search is the first modeled provider-native tool. Other built-in tools are not yet exposed through `ProviderTools`.
- Web Search option parity is intentionally limited to controls the selected provider can enforce. Unsupported portable constraints fail validation instead of being ignored.
- Prompt-cache intent is portable where semantics match, but breakpoint placement and lifecycle are provider-specific. Explicit resource lifecycle is currently Gemini-only.
- Evidence has a portable graph, but provider-required attribution, retrieval/store identifiers, Maps/place details, and native scores may remain provider-specific.
- Schema support and reasoning controls vary by model.
- Provider-native continuation state cannot be moved between providers.
- This layer does not implement a mandatory in-process/distributed limiter, batch requests, embeddings, realtime audio, or media generation.

Unsupported input is rejected rather than silently omitted.

## Testing

The `LLMAbstraction.Tests` project contains deterministic contract tests for:

- Provider request JSON
- Endpoints and authentication headers
- System instruction normalization
- Images and files
- Structured output and reasoning
- Tool calls and tool results
- Mixed function/provider tools and Web Search option translation
- Search calls, results, sources, images, suggestions, and citations
- Streaming SSE lifecycles
- Provider-native continuation round trips
- Cross-provider isolation
- Validation diagnostics
- Provider option collision handling
- Structured API errors
- Retries, cancellation, admission leases, streaming retry boundaries, and rate-limit headers
- Provider token-count endpoint and request-shape parity
- Evidence graph, typed citation locations, grouped grounding, citable input documents, and required attribution artifacts
- Prompt-cache placement, TTL/order diagnostics, unknown-model warnings, usage accounting, token-count parity, endpoint routing, and Gemini resource lifecycle
- Refusal, safety, finish, cache, and usage metadata

Run the suite with:

```bash
dotnet test AgentFlow/AgentFlow.sln
```

No live API credentials are required for the contract suite.
