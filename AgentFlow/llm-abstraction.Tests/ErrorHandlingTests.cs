using System.Net;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Errors;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Transport;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class ErrorHandlingTests
{
    [Fact]
    public async Task OpenAIErrorRetainsStructuredFieldsHeadersAndRetryability()
    {
        const string body = """
        {"error":{"message":"Rate limit reached","type":"rate_limit_error","param":"model","code":"rate_limit_exceeded"}}
        """;
        using var client = Client(body, HttpStatusCode.TooManyRequests, new Dictionary<string, string>
        {
            ["x-request-id"] = "req_openai",
            ["retry-after"] = "5"
        });
        var service = new OpenAIService(client, "key", transportOptions: NoRetries());

        var error = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(Request("gpt-5.6")));

        Assert.Equal(LLMProvider.OpenAI, error.Provider);
        Assert.Equal(HttpStatusCode.TooManyRequests, error.StatusCode);
        Assert.Equal("rate_limit_exceeded", error.ErrorCode);
        Assert.Equal("rate_limit_error", error.ErrorType);
        Assert.Equal("model", error.Parameter);
        Assert.Equal("req_openai", error.RequestId);
        Assert.Equal(TimeSpan.FromSeconds(5), error.RetryAfter);
        Assert.True(error.IsTransient);
        Assert.Contains("Rate limit reached", error.Message);
        Assert.IsAssignableFrom<HttpRequestException>(error);
    }

    [Fact]
    public async Task ClaudeErrorRetainsAnthropicErrorTypeAndRequestId()
    {
        const string body = """
        {"type":"error","error":{"type":"invalid_request_error","message":"Thinking block was modified"}}
        """;
        using var client = Client(body, HttpStatusCode.BadRequest, new Dictionary<string, string>
        {
            ["request-id"] = "req_claude"
        }, "https://api.anthropic.com/");
        var service = new ClaudeService(client, "key");

        var error = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(Request("claude-sonnet-4-6")));

        Assert.Equal(LLMProvider.Claude, error.Provider);
        Assert.Equal("invalid_request_error", error.ErrorType);
        Assert.Equal("req_claude", error.RequestId);
        Assert.False(error.IsTransient);
        Assert.Equal(body, error.RawResponse);
    }

    [Fact]
    public async Task GeminiErrorRetainsGoogleStatusNumericCodeAndDetails()
    {
        const string body = """
        {"error":{"code":400,"message":"Invalid function response","status":"INVALID_ARGUMENT","details":[{"reason":"BAD_TOOL_NAME"}]}}
        """;
        using var client = Client(body, HttpStatusCode.BadRequest, baseAddress: "https://generativelanguage.googleapis.com/v1beta/");
        var service = new GeminiService(client, "key");

        var error = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(Request("gemini-3.5-flash")));

        Assert.Equal(LLMProvider.Gemini, error.Provider);
        Assert.Equal("400", error.ErrorCode);
        Assert.Equal("INVALID_ARGUMENT", error.ErrorType);
        Assert.Equal("BAD_TOOL_NAME", error.Details!.Value[0].GetProperty("reason").GetString());
    }

    [Fact]
    public async Task NonJsonProxyErrorIsStillStructuredAndPreservesRawBody()
    {
        using var client = Client("upstream unavailable", HttpStatusCode.BadGateway);
        var service = new OpenAIService(client, "key", transportOptions: NoRetries());

        var error = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(Request("gpt-5.6")));

        Assert.True(error.IsTransient);
        Assert.Equal("upstream unavailable", error.RawResponse);
        Assert.Contains("502", error.Message);
    }

    private static UnifiedRequest Request(string model) => new()
    {
        Model = model,
        Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
    };

    private static LLMTransportOptions NoRetries() => new()
    {
        Retry = new RetryPolicy { Enabled = false }
    };

    private static HttpClient Client(
        string body,
        HttpStatusCode statusCode,
        IReadOnlyDictionary<string, string>? headers = null,
        string baseAddress = "https://api.openai.com/v1/")
    {
        return new HttpClient(new TestHttpMessageHandler(body, statusCode, headers))
        {
            BaseAddress = new Uri(baseAddress)
        };
    }
}
