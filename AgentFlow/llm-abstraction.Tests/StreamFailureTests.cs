using System.Net;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Errors;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public class StreamFailureTests
{
    [Theory]
    [InlineData(LLMProvider.OpenAI)]
    [InlineData(LLMProvider.Claude)]
    [InlineData(LLMProvider.Gemini)]
    public async Task ProviderErrorAfterOutputPreservesDetails(LLMProvider provider)
    {
        var prefix = provider switch
        {
            LLMProvider.OpenAI => "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
            LLMProvider.Claude => "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            _ => "data: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"Hi\"}}\n\n"
        };
        var error = "event: error\ndata: {\"type\":\"error\",\"event_type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"code\":\"busy\",\"message\":\"Try later\"}}\n\n";
        using var client = new HttpClient(new TestHttpMessageHandler(prefix + error)) { BaseAddress = new Uri("https://example.test/") };
        var service = Create(provider, client);
        var exception = await Assert.ThrowsAsync<LLMApiException>(async () =>
        {
            await foreach (var chunk in service.StreamAsync(Request())) { }
        });
        Assert.Equal("busy", exception.ErrorCode);
        Assert.Equal("Try later", exception.Message);
        Assert.True(exception.HasPartialOutput);
        Assert.False(exception.HasToolCalls);
    }

    [Theory]
    [InlineData(LLMProvider.OpenAI)]
    [InlineData(LLMProvider.Claude)]
    [InlineData(LLMProvider.Gemini)]
    public async Task EmptyStreamIsTransportFailure(LLMProvider provider)
    {
        using var client = new HttpClient(new TestHttpMessageHandler("")) { BaseAddress = new Uri("https://example.test/") };
        var error = await Assert.ThrowsAsync<LLMTransportException>(async () =>
        {
            await foreach (var chunk in Create(provider, client).StreamAsync(Request())) { }
        });
        Assert.False(error.HasPartialOutput);
    }

    [Theory]
    [InlineData(LLMProvider.OpenAI)]
    [InlineData(LLMProvider.Claude)]
    [InlineData(LLMProvider.Gemini)]
    public async Task CancelledEnumerationPreservesCancellation(LLMProvider provider)
    {
        using var client = new HttpClient(new TestHttpMessageHandler("")) { BaseAddress = new Uri("https://example.test/") };
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var chunk in Create(provider, client).StreamAsync(Request(), cancellation.Token)) { }
        });
    }

    private static ILLMService Create(LLMProvider provider, HttpClient client) => provider switch
    {
        LLMProvider.OpenAI => new OpenAIService(client, "key"),
        LLMProvider.Claude => new ClaudeService(client, "key"),
        _ => new GeminiService(client, "key")
    };
    private static UnifiedRequest Request() => new() { Model = "test-model", Messages = { new UnifiedMessage(MessageRole.User, "Hi") } };
}
