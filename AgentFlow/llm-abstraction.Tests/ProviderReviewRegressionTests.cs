using System.Text.Json;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public class ProviderReviewRegressionTests
{
    [Fact]
    public async Task InteractionStreamReconstructsTextWithoutAnInitialContentArray()
    {
        const string events = """
            data: {"event_type":"step.start","index":0,"step":{"type":"model_output"}}

            data: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":"blue"}}

            data: {"event_type":"step.stop","index":0}

            data: {"event_type":"interaction.completed","interaction":{"id":"test","status":"completed"}}

            data: [DONE]
            """;
        using var client = new HttpClient(new TestHttpMessageHandler(events));
        var service = new GeminiService(client, "test-key");
        var chunks = new List<StreamChunk>();
        await foreach (var chunk in service.StreamAsync(Request("gemini-4-hydrogen")))
            chunks.Add(chunk);
        Assert.Equal("blue", Assert.IsType<TextContent>(Assert.Single(chunks.Last().CompletedMessage!.Content)).Text);
    }

    [Fact]
    public void InteractionToolResultAndResponseFormatUseNativeJsonShapes()
    {
        var request = Request("gemini-4-hydrogen");
        request.Output = new() { Kind = OutputFormatKind.JsonObject };
        request.Messages.Add(new(MessageRole.Tool, [new ToolResultContent
        {
            ToolName = "lookup", ToolCallId = "call-1", Output = new { code = 7 }
        }]));
        var payload = new GeminiInteractionsConverter().ConvertRequest(request);
        Assert.Equal("application/json", Assert.Single(payload.ResponseFormat!).GetProperty("mime_type").GetString());
        var value = Assert.Single(payload.Input.Last().GetProperty("result").EnumerateArray());
        Assert.Equal("text", value.GetProperty("type").GetString());
        Assert.Equal("""{"code":7}""", value.GetProperty("text").GetString());
    }

    [Fact]
    public async Task InteractionStreamBuildsToolContinuationFromSteps()
    {
        var events = new[]
        {
            """{"event_type":"step.start","index":0,"step":{"type":"function_call","id":"call1","name":"weather","arguments":{}}}""",
            """{"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"{\"city\":\"Boston\"}"}}""",
            """{"event_type":"step.stop","index":0}""",
            """{"event_type":"interaction.completed","interaction":{"id":"int1","model":"gemini-3.8-flash","status":"requires_action","usage":{"total_input_tokens":4,"total_output_tokens":5,"total_tokens":9}}}""",
            "[DONE]"
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
