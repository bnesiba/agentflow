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
| JSON mode | Supported | Supported | Supported |
| JSON Schema output | Provider schema subset | Provider schema subset | Provider schema subset |
| Reasoning controls | Manual/adaptive subset | Effort/summary subset | Budget/level subset |
| Native reasoning continuation | Preserved | Preserved | Preserved |
| Provider-native Web Search | Supported | Supported | Supported |
| Search calls/results | Portable + native | Portable + native | Portable + native |
| Citations/grounding | Portable + native | Portable + native | Portable + native |

Model capabilities and accepted schema keywords remain model-dependent. The provider API is the final authority for model-specific validation.

Gemini uses the Interactions API by default. A Web Search time-range filter automatically uses the legacy `generateContent` endpoint because that control is not exposed by Interactions.

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

Function tools and provider-native tools share one collection but have different execution ownership. Your application executes a `FunctionTool`; Anthropic, OpenAI, or Gemini executes a `ProviderTool` on its own infrastructure.

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

Provider-managed search activity appears as `ProviderToolCallContent` and `ProviderToolResultContent`; it never appears as `ToolCallContent`, so callers do not accidentally execute it. Results expose portable `WebSource` objects and Gemini search-suggestion markup. Text citations are exposed through `TextContent.Citations`. Every projection also retains its native provider JSON for provider-specific fields and correct continuation.

Applications that display grounded answers remain responsible for rendering the returned citations and any provider-required attribution UI. In particular, preserve and render Gemini's `SearchSuggestionsHtml` when applicable. Anthropic may return `FinishReason.Pause` during a long server-tool turn; continue by replaying the returned assistant message unchanged with the same tool definition.

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
request.ResponseFormat = new ResponseFormat
{
    Type = ResponseFormatType.JsonSchema,
    JsonSchema = new JsonSchema
    {
        Name = "person",
        Strict = true,
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

`ResponseFormatType.Json` requests JSON without a caller-supplied strict schema. `JsonSchema` requires a non-empty schema. Providers support different JSON Schema subsets and may reject overly complex or unsupported keywords.

## Reasoning controls

```csharp
request.Reasoning = new ReasoningOptions
{
    Enabled = true,
    Effort = "medium",
    Summary = "auto",
    IncludeThoughts = true,
    BudgetTokens = null
};
```

These properties are provider-dependent:

- Claude: manual `BudgetTokens`, adaptive effort subset
- OpenAI: `Effort` and `Summary`
- Gemini: `BudgetTokens` or `Effort`/thinking level, plus `IncludeThoughts`

Gemini budget and level controls cannot be supplied together through the portable request. Claude manual budgets must satisfy provider constraints.

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
- Provider-specific reasoning/tool conflicts
- Provider continuation mismatches

Inspect diagnostics without sending a request:

```csharp
using LLMAbstraction.Core.Validation;

var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.OpenAI);
var capabilities = LLMRequestValidator.GetCapabilities(LLMProvider.OpenAI);
```

Converters throw `LLMRequestValidationException` when error diagnostics exist. Warnings remain available through explicit validation.

Capability profiles describe the portable provider-level contract. They do not claim that every model supports every provider feature.

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
}
```

The raw response and parsed provider detail JSON remain available for diagnostics.

## Known portability limits

- Web Search is the first modeled provider-native tool. Other built-in tools are not yet exposed through `ProviderTools`.
- Web Search option parity is intentionally limited to controls the selected provider can enforce. Unsupported portable constraints fail validation instead of being ignored.
- Prompt-cache breakpoints and all cache-control strategies are not portable. Claude cache usage metrics are preserved.
- Provider citations, search results, grounding, refusals, and safety fields do not share one universal schema. Common information is exposed where possible and native metadata is retained.
- Schema support and reasoning controls vary by model.
- Provider-native continuation state cannot be moved between providers.
- This layer does not implement retries, rate limiting, batch requests, token counting, embeddings, realtime audio, or media generation.

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
- Refusal, safety, finish, cache, and usage metadata

Run the suite with:

```bash
dotnet test AgentFlow/AgentFlow.sln
```

No live API credentials are required for the contract suite.
