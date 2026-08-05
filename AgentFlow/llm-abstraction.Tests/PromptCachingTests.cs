using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Claude.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.Gemini.Models;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.OpenAI.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class PromptCachingTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    [Fact]
    public void AnthropicSerializesAutomaticAndExplicitCacheControls()
    {
        var request = Request("claude-sonnet-4-6");
        request.Instructions = "Stable policy";
        request.Cache = new PromptCacheOptions
        {
            Mode = PromptCacheMode.PreferReuse,
            Ttl = PromptCacheTtl.OneHour
        };
        request.InstructionsCache = new PromptCacheDirective { Ttl = PromptCacheTtl.OneHour };
        request.Messages[0].Content[0].Cache = new PromptCacheDirective();
        request.Tools = new ToolCollection
        {
            new FunctionTool
            {
                Id = "lookup",
                Name = "lookup",
                Description = "Look up a record",
                Parameters = ObjectSchema(),
                Cache = new PromptCacheDirective { Ttl = PromptCacheTtl.OneHour }
            }
        };

        var json = Serialize(new ClaudeConverter().ConvertRequest(request));

        Assert.Equal("1h", json["cache_control"]!["ttl"]!.GetValue<string>());
        Assert.Equal("1h", json["tools"]![0]!["cache_control"]!["ttl"]!.GetValue<string>());
        Assert.Equal("1h", json["system"]![0]!["cache_control"]!["ttl"]!.GetValue<string>());
        Assert.Equal("ephemeral", json["messages"]![0]!["content"]![0]!["cache_control"]!["type"]!.GetValue<string>());
        Assert.Null(json["messages"]![0]!["content"]![0]!["cache_control"]!["ttl"]);
    }

    [Fact]
    public void AnthropicRejectsInvalidTtlOrderAndCountsAutomaticBreakpoint()
    {
        var request = Request("future-claude-model");
        request.Cache = new PromptCacheOptions { Mode = PromptCacheMode.PreferReuse };
        request.Messages[0].Content = new List<ContentBlock>
        {
            new TextContent { Text = "a", Cache = new PromptCacheDirective() },
            new TextContent { Text = "b", Cache = new PromptCacheDirective { Ttl = PromptCacheTtl.OneHour } },
            new TextContent { Text = "c", Cache = new PromptCacheDirective() },
            new TextContent { Text = "d", Cache = new PromptCacheDirective() }
        };

        var diagnostics = LLMRequestValidator.Validate(
            request, LLMProvider.Claude, LLMApiSurface.AnthropicMessages);

        Assert.Contains(diagnostics, item => item.Code == "claude.cache.ttl_order" && item.Severity == RequestDiagnosticSeverity.Error);
        Assert.Contains(diagnostics, item => item.Code == "claude.cache.breakpoint_limit" && item.Severity == RequestDiagnosticSeverity.Error);
        Assert.Contains(diagnostics, item => item.Code == "request.cache.unknown_model" && item.Severity == RequestDiagnosticSeverity.Warning);
    }

    [Fact]
    public void OpenAISerializesExplicitBreakpointsAndRequestOptionsForUnknownModel()
    {
        var request = Request("gpt-future-99");
        request.Cache = new PromptCacheOptions
        {
            Mode = PromptCacheMode.ExplicitBreakpointsOnly,
            Ttl = PromptCacheTtl.ThirtyMinutes,
            OpenAI = new OpenAIPromptCacheOptions { CacheKey = "tenant-42" }
        };
        request.Messages[0].Content[0].Cache = new PromptCacheDirective();

        var diagnostics = LLMRequestValidator.Validate(
            request, LLMProvider.OpenAI, LLMApiSurface.OpenAIResponses);
        var json = Serialize(new OpenAIConverter().ConvertRequest(request));

        Assert.DoesNotContain(diagnostics, item => item.Severity == RequestDiagnosticSeverity.Error);
        Assert.Contains(diagnostics, item => item.Code == "request.cache.unknown_model");
        Assert.Equal("tenant-42", json["prompt_cache_key"]!.GetValue<string>());
        Assert.Equal("explicit", json["prompt_cache_options"]!["mode"]!.GetValue<string>());
        Assert.Equal("30m", json["prompt_cache_options"]!["ttl"]!.GetValue<string>());
        Assert.Equal("explicit", json["input"]![0]!["content"]![0]!["prompt_cache_breakpoint"]!["mode"]!.GetValue<string>());
    }

    [Fact]
    public void OpenAIRejectsOnlyUnrepresentableBreakpointPlacements()
    {
        var request = Request("gpt-future-99");
        request.InstructionsCache = new PromptCacheDirective();
        request.Messages[0].Content.Add(new ToolResultContent
        {
            ToolCallId = "call-1",
            Output = "done",
            Cache = new PromptCacheDirective()
        });

        var diagnostics = LLMRequestValidator.Validate(
            request, LLMProvider.OpenAI, LLMApiSurface.OpenAIResponses);

        Assert.Contains(diagnostics, item => item.Code == "openai.cache.instructions_breakpoint_unrepresentable");
        Assert.Contains(diagnostics, item => item.Code == "openai.cache.breakpoint_unrepresentable");
    }

    [Fact]
    public void GeminiGenerateContentReferencesExplicitCacheResource()
    {
        var request = Request("gemini-future-model");
        request.Cache = new PromptCacheOptions
        {
            Mode = PromptCacheMode.PreferReuse,
            Gemini = new GeminiPromptCacheOptions { CachedContentName = "cachedContents/cache-123" }
        };

        var json = Serialize(new GeminiConverter().ConvertRequest(request));

        Assert.Equal("cachedContents/cache-123", json["cachedContent"]!.GetValue<string>());
    }

    [Fact]
    public void GeminiInteractionsRejectsExplicitResourceWithoutBlockingUnknownImplicitCaching()
    {
        var request = Request("gemini-future-model");
        request.Cache = new PromptCacheOptions
        {
            Mode = PromptCacheMode.PreferReuse,
            Gemini = new GeminiPromptCacheOptions { CachedContentName = "cachedContents/cache-123" }
        };

        var diagnostics = LLMRequestValidator.Validate(
            request, LLMProvider.Gemini, LLMApiSurface.GeminiInteractions);

        Assert.Contains(diagnostics, item => item.Code == "gemini.interactions.explicit_cache_unrepresentable");
        Assert.Contains(diagnostics, item => item.Code == "request.cache.unknown_model" && item.Severity == RequestDiagnosticSeverity.Warning);
    }

    [Fact]
    public void ProviderCacheUsageIsNormalized()
    {
        var claude = JsonSerializer.Deserialize<ClaudeMessageResponse>(
            """{"id":"m","model":"claude","content":[],"usage":{"input_tokens":100,"output_tokens":10,"cache_creation_input_tokens":80,"cache_read_input_tokens":60,"cache_creation":{"ephemeral_5m_input_tokens":30,"ephemeral_1h_input_tokens":50}}}""")!;
        var openAI = JsonSerializer.Deserialize<OpenAIResponse>(
            """{"id":"r","model":"gpt","output":[],"usage":{"input_tokens":100,"output_tokens":10,"total_tokens":110,"input_tokens_details":{"cached_tokens":60,"cache_write_tokens":40}}}""")!;
        var gemini = JsonSerializer.Deserialize<GeminiGenerateResponse>(
            """{"responseId":"g","modelVersion":"gemini","candidates":[],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":10,"totalTokenCount":110,"cachedContentTokenCount":70}}""")!;

        var claudeUsage = new ClaudeConverter().ConvertResponse(claude).Usage;
        var openAIUsage = new OpenAIConverter().ConvertResponse(openAI).Usage;
        var geminiUsage = new GeminiConverter().ConvertResponse(gemini).Usage;

        Assert.Equal(80, claudeUsage.CacheWriteTokens);
        Assert.Equal(30, claudeUsage.CacheWrite5MinuteTokens);
        Assert.Equal(50, claudeUsage.CacheWrite1HourTokens);
        Assert.Equal(60, openAIUsage.CacheReadTokens);
        Assert.Equal(40, openAIUsage.CacheWriteTokens);
        Assert.Equal(70, geminiUsage.CacheReadTokens);
    }

    [Fact]
    public async Task GeminiInteractionsModeRoutesCachedContentGenerationToGenerateContent()
    {
        var handler = new TestHttpMessageHandler(
            """{"responseId":"g","modelVersion":"gemini-future-model","candidates":[],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":0,"totalTokenCount":1}}""");
        using var client = new HttpClient(handler) { BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/") };
        var service = new GeminiService(client, "key", apiMode: GeminiApiMode.Interactions);
        var request = Request("gemini-future-model");
        request.Cache = new PromptCacheOptions
        {
            Mode = PromptCacheMode.PreferReuse,
            Gemini = new GeminiPromptCacheOptions { CachedContentName = "cachedContents/cache-123" }
        };

        await service.GenerateAsync(request);

        Assert.EndsWith("/models/gemini-future-model:generateContent", handler.LastRequest!.RequestUri!.AbsoluteUri);
        Assert.Contains("\"cachedContent\":\"cachedContents/cache-123\"", handler.LastRequestBody);
    }

    [Fact]
    public async Task GeminiExplicitCacheCreateUsesConvertedPortablePrefix()
    {
        var response = """{"name":"cachedContents/cache-123","model":"models/gemini-future-model","displayName":"policy","createTime":"2026-08-05T12:00:00Z","expireTime":"2026-08-05T13:00:00Z","usageMetadata":{"totalTokenCount":44}}""";
        var handler = new TestHttpMessageHandler(response);
        using var client = new HttpClient(handler) { BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/") };
        var service = new GeminiService(client, "key");
        var prefix = Request("gemini-future-model");
        prefix.Instructions = "Stable policy";

        var resource = await service.CreatePromptCacheAsync(new PromptCacheCreateRequest
        {
            Prefix = prefix,
            DisplayName = "policy",
            Ttl = TimeSpan.FromMinutes(10)
        });
        var body = JsonNode.Parse(handler.LastRequestBody!)!;

        Assert.Equal(HttpMethod.Post, handler.LastRequest!.Method);
        Assert.EndsWith("/cachedContents", handler.LastRequest.RequestUri!.AbsoluteUri);
        Assert.Equal("models/gemini-future-model", body["model"]!.GetValue<string>());
        Assert.Equal("600s", body["ttl"]!.GetValue<string>());
        Assert.Equal("Stable policy", body["systemInstruction"]!["parts"]![0]!["text"]!.GetValue<string>());
        Assert.Equal("hello", body["contents"]![0]!["parts"]![0]!["text"]!.GetValue<string>());
        Assert.Equal("cachedContents/cache-123", resource.Name);
        Assert.Equal(44, resource.CachedTokens);
        Assert.NotNull(resource.Transport);
    }

    [Fact]
    public async Task GeminiExplicitCacheGetUpdateListAndDeleteUseDocumentedResourceEndpoints()
    {
        const string resourceJson = """{"name":"cachedContents/cache-123","model":"models/gemini-future-model","expireTime":"2026-08-05T13:00:00Z"}""";

        var getHandler = new TestHttpMessageHandler(resourceJson);
        using (var client = new HttpClient(getHandler) { BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/") })
        {
            await new GeminiService(client, "key").GetPromptCacheAsync("cachedContents/cache-123");
            Assert.Equal(HttpMethod.Get, getHandler.LastRequest!.Method);
            Assert.EndsWith("/cachedContents/cache-123", getHandler.LastRequest.RequestUri!.AbsoluteUri);
        }

        var patchHandler = new TestHttpMessageHandler(resourceJson);
        using (var client = new HttpClient(patchHandler) { BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/") })
        {
            await new GeminiService(client, "key").UpdatePromptCacheExpirationAsync(
                "cachedContents/cache-123",
                new PromptCacheExpiration { Ttl = TimeSpan.FromHours(2) });
            Assert.Equal(HttpMethod.Patch, patchHandler.LastRequest!.Method);
            Assert.EndsWith("/cachedContents/cache-123?updateMask=ttl", patchHandler.LastRequest.RequestUri!.AbsoluteUri);
            Assert.Equal("7200s", JsonNode.Parse(patchHandler.LastRequestBody!)!["ttl"]!.GetValue<string>());
        }

        var listHandler = new TestHttpMessageHandler(
            $$"""{"cachedContents":[{{resourceJson}}],"nextPageToken":"next"}""");
        using (var client = new HttpClient(listHandler) { BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/") })
        {
            var page = await new GeminiService(client, "key").ListPromptCachesAsync(25, "a+b");
            Assert.Single(page.Items);
            Assert.Equal("next", page.NextPageToken);
            Assert.EndsWith("/cachedContents?pageSize=25&pageToken=a%2Bb", listHandler.LastRequest!.RequestUri!.AbsoluteUri);
        }

        var deleteHandler = new TestHttpMessageHandler("{}");
        using (var client = new HttpClient(deleteHandler) { BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/") })
        {
            await new GeminiService(client, "key").DeletePromptCacheAsync("cachedContents/cache-123");
            Assert.Equal(HttpMethod.Delete, deleteHandler.LastRequest!.Method);
        }
    }

    [Fact]
    public async Task ProviderTokenCountPayloadsRetainPromptCacheConfiguration()
    {
        var openAIHandler = new TestHttpMessageHandler("""{"input_tokens":12}""");
        using (var client = new HttpClient(openAIHandler) { BaseAddress = new Uri("https://api.openai.com/v1/") })
        {
            var request = Request("gpt-future-99");
            request.Cache = new PromptCacheOptions { Mode = PromptCacheMode.ExplicitBreakpointsOnly };
            request.Messages[0].Content[0].Cache = new PromptCacheDirective();
            await new OpenAIService(client, "key").CountInputTokensAsync(request);
            Assert.Contains("prompt_cache_breakpoint", openAIHandler.LastRequestBody);
            Assert.Contains("prompt_cache_options", openAIHandler.LastRequestBody);
        }

        var claudeHandler = new TestHttpMessageHandler("""{"input_tokens":12}""");
        using (var client = new HttpClient(claudeHandler) { BaseAddress = new Uri("https://api.anthropic.com/v1/") })
        {
            var request = Request("future-claude-model");
            request.Cache = new PromptCacheOptions { Mode = PromptCacheMode.PreferReuse };
            await new ClaudeService(client, "key").CountInputTokensAsync(request);
            Assert.Contains("cache_control", claudeHandler.LastRequestBody);
        }

        var geminiHandler = new TestHttpMessageHandler("""{"totalTokens":12,"cachedContentTokenCount":8}""");
        using (var client = new HttpClient(geminiHandler) { BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/") })
        {
            var request = Request("gemini-future-model");
            request.Cache = new PromptCacheOptions
            {
                Gemini = new GeminiPromptCacheOptions { CachedContentName = "cachedContents/cache-123" }
            };
            var count = await new GeminiService(client, "key").CountInputTokensAsync(request);
            Assert.Contains("cachedContent", geminiHandler.LastRequestBody);
            Assert.Equal(8, count.CachedTokens);
        }
    }

    private static UnifiedRequest Request(string model) => new()
    {
        Model = model,
        Messages = new List<UnifiedMessage>
        {
            new(MessageRole.User, new List<ContentBlock> { new TextContent { Text = "hello" } })
        }
    };

    private static Dictionary<string, object> ObjectSchema() => new()
    {
        ["type"] = "object",
        ["properties"] = new Dictionary<string, object>()
    };

    private static JsonNode Serialize<T>(T value) =>
        JsonNode.Parse(JsonSerializer.Serialize(value, JsonOptions))!;
}
