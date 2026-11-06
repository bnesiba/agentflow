using System.Text.Json;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public class ProviderReviewRegressionTests
{
    [Fact]
    public async Task InteractionStreamBuildsToolContinuationFromSteps()
    {
        var events = new[]
        {
            """{"event_type":"step.start","index":0,"step":{"type":"function_call","id":"call1","name":"weather","arguments":{}}}""",
            """{"event_type":"step.delta","index":0,"delta":{"type":"function_call","arguments_delta":"{\"city\":\"Boston\"}"}}""",
            """{"event_type":"step.stop","index":0}""",
            """{"event_type":"interaction.completed","interaction":{"id":"int1","model":"gemini-3.8-flash","status":"requires_action","usage":{"total_input_tokens":4,"total_output_tokens":5,"total_tokens":9}}}"""
        };
        using var client = new HttpClient(new TestHttpMessageHandler(string.Join("\n\n", events.Select(value => "data: " + value)) + "\n\n"));
        var service = new GeminiService(client, "test-key");
        var chunks = new List<StreamChunk>();
        await foreach (var chunk in service.StreamAsync(Request("gemini-3.8-flash")))
            chunks.Add(chunk);
        var final = chunks.Last();
        var call = Assert.IsType<ToolCallContent>(Assert.Single(final.CompletedMessage!.Content));
        Assert.Equal("call1", call.Id);
        Assert.Equal("Boston", ((JsonElement)call.Input["city"]).GetString());
        Assert.Single(final.Continuation!.NativeItems);
        Assert.Equal(FinishReason.ToolCalls, final.FinishReason);
    }

    [Fact]
    public void EndpointSwitchCannotDiscardInteractionHistory()
    {
        var request = Request("gemini-3.8-flash");
        request.Continuation = new ProviderContinuationState { Provider = ProviderIds.Gemini, ResponseId = "int1", Mode = ContinuationMode.ServerManaged };
        Assert.Throws<NotSupportedException>(() => new GeminiConverter().ConvertRequest(request));
    }

    [Fact]
    public void OrdinaryToolSchemaKeepsOptionalProperties()
    {
        var request = Request("gpt-5-mini");
        request.Tools = new ToolCollection { new FunctionTool { Name = "weather", Parameters = new() { ["type"] = "object" } } };
        var converted = new OpenAIConverter().ConvertRequest(request);
        Assert.False(converted.Tools![0].Strict);
    }

    private static UnifiedRequest Request(string model) => new()
    {
        Model = model,
        Messages = { new UnifiedMessage(MessageRole.User, "Weather?") }
    };
}
