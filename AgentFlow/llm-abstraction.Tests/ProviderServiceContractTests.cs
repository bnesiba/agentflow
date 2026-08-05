using System.Net.Http.Headers;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class ProviderServiceContractTests
{
    [Fact]
    public async Task ClaudeUsesMessagesEndpointAndRequiredHeaders()
    {
        const string response = """
        {"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[{"type":"text","text":"Hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}
        """;
        var handler = new TestHttpMessageHandler(response);
        using var client = new HttpClient(handler) { BaseAddress = new Uri("https://api.anthropic.com/") };
        var service = new ClaudeService(client, "claude-key", "2023-06-01");

        await service.GenerateAsync(Request("claude-sonnet-4-6"));

        Assert.Equal("https://api.anthropic.com/v1/messages", handler.LastRequest!.RequestUri!.ToString());
        Assert.Equal("claude-key", Assert.Single(handler.LastRequest.Headers.GetValues("x-api-key")));
        Assert.Equal("2023-06-01", Assert.Single(handler.LastRequest.Headers.GetValues("anthropic-version")));
        Assert.Contains("\"max_tokens\":1024", handler.LastRequestBody);
    }

    [Fact]
    public async Task OpenAIUsesResponsesEndpointAndBearerAuthentication()
    {
        const string response = """
        {"id":"resp_1","model":"gpt-5.6","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}
        """;
        var handler = new TestHttpMessageHandler(response);
        using var client = new HttpClient(handler) { BaseAddress = new Uri("https://api.openai.com/v1/") };
        var service = new OpenAIService(client, "openai-key");

        await service.GenerateAsync(Request("gpt-5.6"));

        Assert.Equal("https://api.openai.com/v1/responses", handler.LastRequest!.RequestUri!.ToString());
        Assert.Equal(new AuthenticationHeaderValue("Bearer", "openai-key"), handler.LastRequest.Headers.Authorization);
        Assert.Contains("\"input\"", handler.LastRequestBody);
    }

    [Fact]
    public async Task GeminiUsesGenerateContentEndpointAndApiKeyHeaderNotQueryString()
    {
        const string response = """
        {"responseId":"gem_1","modelVersion":"gemini-3.5-flash","candidates":[{"index":0,"finishReason":"STOP","content":{"role":"model","parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}
        """;
        var handler = new TestHttpMessageHandler(response);
        using var client = new HttpClient(handler)
        {
            BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/")
        };
        var service = new GeminiService(client, "gemini-key");

        await service.GenerateAsync(Request("gemini-3.5-flash"));

        Assert.Equal(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
            handler.LastRequest!.RequestUri!.ToString());
        Assert.DoesNotContain("gemini-key", handler.LastRequest.RequestUri.Query);
        Assert.Equal("gemini-key", Assert.Single(handler.LastRequest.Headers.GetValues("x-goog-api-key")));
    }

    private static UnifiedRequest Request(string model) => new()
    {
        Model = model,
        Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
    };
}
