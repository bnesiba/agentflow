using System.Text.Json;
using System.Text.Json.Nodes;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class StreamingToolCallTests
{
    [Fact]
    public async Task ClaudeStreamAccumulatesSignedThinkingToolArgumentsAndUsage()
    {
        var sse = BuildClaudeEvent("message_start", """
        {"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"usage":{"input_tokens":11,"output_tokens":0,"cache_read_input_tokens":3}}}
        """) +
        BuildClaudeEvent("content_block_start", """
        {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}
        """) +
        BuildClaudeEvent("content_block_delta", """
        {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Need weather."}}
        """) +
        BuildClaudeEvent("content_block_delta", """
        {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed-thinking"}}
        """) +
        BuildClaudeEvent("content_block_stop", """
        {"type":"content_block_stop","index":0}
        """) +
        BuildClaudeEvent("content_block_start", """
        {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_stream","name":"get_weather","input":{}}}
        """) +
        BuildClaudeEvent("content_block_delta", """
        {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}
        """) +
        BuildClaudeEvent("content_block_delta", """
        {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"Boston\"}"}}
        """) +
        BuildClaudeEvent("content_block_stop", """
        {"type":"content_block_stop","index":1}
        """) +
        BuildClaudeEvent("message_delta", """
        {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":17}}
        """);

        using var client = new HttpClient(new TestHttpMessageHandler(sse))
        {
            BaseAddress = new Uri("https://api.anthropic.com/")
        };
        var service = new ClaudeService(client, "test-key");
        var chunks = await CollectAsync(service.StreamAsync(new UnifiedRequest
        {
            Model = "claude-sonnet-4-6",
            Messages = { new UnifiedMessage(MessageRole.User, "Weather?") }
        }));

        var toolEvents = chunks
            .SelectMany(chunk => chunk.Delta.ToolCalls ?? Enumerable.Empty<ToolCallDelta>())
            .ToList();
        Assert.Equal("toolu_stream", toolEvents[0].Id);
        Assert.Equal("{\"city\":", toolEvents[1].Arguments);
        Assert.Equal("\"Boston\"}", toolEvents[2].Arguments);
        Assert.True(toolEvents[3].IsComplete);
        Assert.Equal("{\"city\":\"Boston\"}", toolEvents[3].CompleteArguments);

        var final = chunks.Last(chunk => chunk.FinishReason != null);
        Assert.Equal(FinishReason.ToolCalls, final.FinishReason);
        Assert.Equal(11, final.Usage!.InputTokens);
        Assert.Equal(17, final.Usage.OutputTokens);
        Assert.Equal(3, final.Usage.CacheReadTokens);
        Assert.Collection(
            final.CompletedMessage!.Content,
            block =>
            {
                var native = Assert.IsType<ProviderNativeContent>(block);
                Assert.Equal("signed-thinking", native.NativeRepresentation!.Value.GetProperty("signature").GetString());
            },
            block =>
            {
                var call = Assert.IsType<ToolCallContent>(block);
                Assert.Equal("toolu_stream", call.Id);
                Assert.Equal("Boston", ((JsonElement)call.Input["city"]).GetString());
            });
    }

    [Fact]
    public async Task OpenAIStreamUsesCallIdAndKeepsItemIdSeparate()
    {
        object[] events =
        {
            new { type = "response.created", response = new { id = "resp_stream", model = "gpt-5.6", status = "in_progress", output = Array.Empty<object>() } },
            new { type = "response.output_item.added", response_id = "resp_stream", output_index = 0, item = new { id = "fc_item", type = "function_call", status = "in_progress", call_id = "call_actual", name = "get_weather", arguments = "" } },
            new { type = "response.function_call_arguments.delta", response_id = "resp_stream", item_id = "fc_item", output_index = 0, delta = "{\"city\":" },
            new { type = "response.function_call_arguments.delta", response_id = "resp_stream", item_id = "fc_item", output_index = 0, delta = "\"Boston\"}" },
            new { type = "response.function_call_arguments.done", response_id = "resp_stream", item_id = "fc_item", output_index = 0, arguments = "{\"city\":\"Boston\"}" },
            new { type = "response.completed", response = new { id = "resp_stream", model = "gpt-5.6", status = "completed", output = new[] { new { id = "fc_item", type = "function_call", status = "completed", call_id = "call_actual", name = "get_weather", arguments = "{\"city\":\"Boston\"}" } }, usage = new { input_tokens = 2, output_tokens = 3, total_tokens = 5 } } }
        };
        var sse = string.Concat(events.Select(value => $"data: {JsonSerializer.Serialize(value)}\n\n"));
        using var client = new HttpClient(new TestHttpMessageHandler(sse))
        {
            BaseAddress = new Uri("https://api.openai.com/v1/")
        };
        var service = new OpenAIService(client, "test-key");
        var chunks = await CollectAsync(service.StreamAsync(new UnifiedRequest
        {
            Model = "gpt-5.6",
            Messages = { new UnifiedMessage(MessageRole.User, "Weather?") }
        }));

        var calls = chunks.SelectMany(chunk => chunk.Delta.ToolCalls ?? []).ToList();
        Assert.All(calls, call => Assert.Equal("call_actual", call.Id));
        Assert.All(calls, call => Assert.Equal("fc_item", call.ItemId));
        Assert.Equal(new[] { "{\"city\":", "\"Boston\"}" },
            calls.Where(call => call.Arguments != null).Select(call => call.Arguments));
        var completed = Assert.Single(calls, call => call.IsComplete);
        Assert.Equal("{\"city\":\"Boston\"}", completed.CompleteArguments);

        var final = chunks.Last();
        Assert.Equal(FinishReason.ToolCalls, final.FinishReason);
        Assert.Equal("call_actual", Assert.IsType<ToolCallContent>(
            Assert.Single(final.CompletedMessage!.Content)).Id);
    }

    [Fact]
    public async Task GeminiStreamReturnsEveryPartToolCallAndReplayableCompletedMessage()
    {
        var first = """
        {"responseId":"gem_stream","modelVersion":"gemini-3.5-flash","candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"internal","thought":true,"thoughtSignature":"sig-thought"},{"text":"Checking ","thoughtSignature":"sig-text"}]}}]}
        """;
        var second = """
        {"responseId":"gem_stream","modelVersion":"gemini-3.5-flash","candidates":[{"index":0,"finishReason":"STOP","content":{"role":"model","parts":[{"text":"now."},{"functionCall":{"id":"gem-call","name":"get_weather","args":{"city":"Boston"}},"thoughtSignature":"sig-call"}]}}],"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":6,"totalTokenCount":10,"thoughtsTokenCount":2}}
        """;
        var sse = $"data: {Compact(first)}\n\ndata: {Compact(second)}\n\n";
        using var client = new HttpClient(new TestHttpMessageHandler(sse))
        {
            BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/")
        };
        var service = new GeminiService(client, "test-key");
        var chunks = await CollectAsync(service.StreamAsync(new UnifiedRequest
        {
            Model = "gemini-3.5-flash",
            Messages = { new UnifiedMessage(MessageRole.User, "Weather?") }
        }));

        Assert.Equal("Checking ", chunks[0].Delta.Content);
        Assert.Equal("now.", chunks[1].Delta.Content);
        var tool = Assert.Single(chunks[1].Delta.ToolCalls!);
        Assert.Equal("gem-call", tool.Id);
        Assert.Equal("get_weather", tool.Name);
        Assert.True(tool.IsComplete);
        Assert.Equal("{\"city\":\"Boston\"}", tool.CompleteArguments);
        Assert.Equal(FinishReason.ToolCalls, chunks[1].FinishReason);
        Assert.Equal("gem_stream", chunks[1].Id);

        Assert.Collection(
            chunks[1].CompletedMessage!.Content,
            block => Assert.IsType<ProviderNativeContent>(block),
            block => Assert.Equal("Checking ", Assert.IsType<TextContent>(block).Text),
            block => Assert.Equal("now.", Assert.IsType<TextContent>(block).Text),
            block => Assert.Equal("gem-call", Assert.IsType<ToolCallContent>(block).Id));
    }

    [Fact]
    public async Task MalformedProviderStreamDataRaisesAnExplicitParsingError()
    {
        const string malformed = "data: {not-json}\n\n";

        using var openAIClient = new HttpClient(new TestHttpMessageHandler(malformed))
        {
            BaseAddress = new Uri("https://api.openai.com/v1/")
        };
        var openAI = new OpenAIService(openAIClient, "key");
        var openAIError = await Assert.ThrowsAsync<InvalidOperationException>(
            () => CollectAsync(openAI.StreamAsync(new UnifiedRequest
            {
                Model = "gpt-5.6",
                Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
            })));
        Assert.Contains("OpenAI", openAIError.Message);

        using var geminiClient = new HttpClient(new TestHttpMessageHandler(malformed))
        {
            BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/")
        };
        var gemini = new GeminiService(geminiClient, "key");
        var geminiError = await Assert.ThrowsAsync<InvalidOperationException>(
            () => CollectAsync(gemini.StreamAsync(new UnifiedRequest
            {
                Model = "gemini-3.5-flash",
                Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
            })));
        Assert.Contains("Gemini", geminiError.Message);
    }

    private static string BuildClaudeEvent(string eventType, string json)
        => $"event: {eventType}\ndata: {Compact(json)}\n\n";

    private static string Compact(string json)
        => JsonNode.Parse(json)!.ToJsonString();

    private static async Task<List<StreamChunk>> CollectAsync(IAsyncEnumerable<StreamChunk> stream)
    {
        var chunks = new List<StreamChunk>();
        await foreach (var chunk in stream)
            chunks.Add(chunk);
        return chunks;
    }
}
