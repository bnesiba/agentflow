using System.Text.Json.Nodes;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class TokenCountingTests
{
    [Fact]
    public async Task OpenAICountsTheConvertedResponsesInputShape()
    {
        var handler = new TestHttpMessageHandler(
            """{"object":"response.input_tokens","input_tokens":321}""",
            headers: new Dictionary<string, string>
            {
                ["x-request-id"] = "req_count_openai",
                ["x-ratelimit-remaining-tokens"] = "8765"
            });
        using var client = new HttpClient(handler) { BaseAddress = new Uri("https://api.openai.com/v1/") };
        var service = new OpenAIService(client, "key");
        var request = AdvancedRequest("future-openai-model");
        request.Parameters = new GenerationParameters
        {
            MaxOutputTokens = 512,
            Temperature = 0.4,
            TopP = 0.9,
            Stream = true
        };
        request.Reasoning = new ReasoningOptions
        {
            Effort = ReasoningEffort.High,
            Output = ReasoningOutput.Summary
        };

        var result = await service.CountInputTokensAsync(request);
        var json = JsonNode.Parse(handler.LastRequestBody!)!.AsObject();

        Assert.EndsWith("/v1/responses/input_tokens", handler.LastRequest!.RequestUri!.AbsoluteUri);
        Assert.Equal("future-openai-model", json["model"]!.GetValue<string>());
        Assert.NotNull(json["input"]);
        Assert.Equal("Be concise.", json["instructions"]!.GetValue<string>());
        Assert.Equal("function", json["tools"]![0]!["type"]!.GetValue<string>());
        Assert.Equal("json_schema", json["text"]!["format"]!["type"]!.GetValue<string>());
        Assert.Equal("high", json["reasoning"]!["effort"]!.GetValue<string>());
        Assert.Null(json["max_output_tokens"]);
        Assert.Null(json["temperature"]);
        Assert.Null(json["top_p"]);
        Assert.Null(json["stream"]);
        Assert.Equal(321, result.InputTokens);
        Assert.Equal(TokenCountAccuracy.Exact, result.Accuracy);
        Assert.Equal(LLMProvider.OpenAI, result.Provider);
        Assert.Equal("req_count_openai", result.Transport!.RequestId);
        Assert.Equal(8765, result.Transport.RateLimits!.Tokens!.Remaining);
        Assert.Equal("response.input_tokens", result.ProviderMetadata["openai.object"]);
    }

    [Fact]
    public async Task ClaudeCountsInputFieldsButOmitsGenerationOnlyFields()
    {
        var handler = new TestHttpMessageHandler(
            """{"input_tokens":222}""",
            headers: new Dictionary<string, string> { ["request-id"] = "req_count_claude" });
        using var client = new HttpClient(handler) { BaseAddress = new Uri("https://api.anthropic.com/") };
        var service = new ClaudeService(client, "key");
        var request = AdvancedRequest("claude-sonnet-4-6");
        request.Parameters = new GenerationParameters
        {
            MaxOutputTokens = 1024,
            TopP = 0.95,
            StopSequences = new List<string> { "END" },
            Stream = true
        };
        request.Reasoning = new ReasoningOptions
        {
            Effort = ReasoningEffort.Medium,
            Output = ReasoningOutput.Summary,
            Anthropic = new AnthropicReasoningOptions { Mode = AnthropicThinkingMode.Adaptive }
        };

        var result = await service.CountInputTokensAsync(request);
        var json = JsonNode.Parse(handler.LastRequestBody!)!.AsObject();

        Assert.EndsWith("/v1/messages/count_tokens", handler.LastRequest!.RequestUri!.AbsoluteUri);
        Assert.Equal("Be concise.", json["system"]!.GetValue<string>());
        Assert.Equal("get_weather", json["tools"]![0]!["name"]!.GetValue<string>());
        Assert.Equal("json_schema", json["output_config"]!["format"]!["type"]!.GetValue<string>());
        Assert.Equal("adaptive", json["thinking"]!["type"]!.GetValue<string>());
        Assert.Null(json["max_tokens"]);
        Assert.Null(json["temperature"]);
        Assert.Null(json["top_p"]);
        Assert.Null(json["stop_sequences"]);
        Assert.Null(json["stream"]);
        Assert.Equal(222, result.InputTokens);
        Assert.Equal(TokenCountAccuracy.Estimate, result.Accuracy);
        Assert.Equal("req_count_claude", result.Transport!.RequestId);
    }

    [Fact]
    public async Task GeminiCountsAFullConvertedGenerateContentRequest()
    {
        const string response = """
        {"totalTokens":444,"cachedContentTokenCount":40,"promptTokensDetails":[{"modality":"TEXT","tokenCount":404}],"cacheTokensDetails":[{"modality":"TEXT","tokenCount":40}]}
        """;
        var handler = new TestHttpMessageHandler(
            response,
            headers: new Dictionary<string, string> { ["x-goog-request-id"] = "req_count_gemini" });
        using var client = new HttpClient(handler)
        {
            BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/")
        };
        var service = new GeminiService(client, "key");
        var request = AdvancedRequest("gemini-3.5-flash");
        request.Reasoning = new ReasoningOptions
        {
            Effort = ReasoningEffort.Medium,
            Output = ReasoningOutput.Summary
        };

        var result = await service.CountInputTokensAsync(request);
        var json = JsonNode.Parse(handler.LastRequestBody!)!.AsObject();
        var converted = json["generateContentRequest"]!;

        Assert.EndsWith("models/gemini-3.5-flash:countTokens", handler.LastRequest!.RequestUri!.AbsoluteUri);
        Assert.Equal("Be concise.", converted["systemInstruction"]!["parts"]![0]!["text"]!.GetValue<string>());
        Assert.Equal("get_weather",
            converted["tools"]![0]!["functionDeclarations"]![0]!["name"]!.GetValue<string>());
        Assert.Equal("application/json", converted["generationConfig"]!["responseMimeType"]!.GetValue<string>());
        Assert.Equal("MEDIUM",
            converted["generationConfig"]!["thinkingConfig"]!["thinkingLevel"]!.GetValue<string>());
        Assert.Equal(444, result.InputTokens);
        Assert.Equal(40, result.CachedTokens);
        Assert.Equal(TokenCountAccuracy.Estimate, result.Accuracy);
        Assert.Equal("req_count_gemini", result.Transport!.RequestId);
        Assert.Equal("generateContent", result.ProviderMetadata["gemini.countSurface"]);
        Assert.Equal("interactions", result.ProviderMetadata["gemini.generationSurface"]);
        Assert.True(result.ProviderMetadata.ContainsKey("gemini.promptTokensDetails"));
        Assert.True(result.ProviderMetadata.ContainsKey("gemini.cacheTokensDetails"));
    }

    [Fact]
    public void FactoryReturnsGenerationAndTokenCountingService()
    {
        ILLMServiceWithTokenCounting service = LLMServiceFactory.CreateOpenAI("key");

        Assert.IsAssignableFrom<ILLMService>(service);
        Assert.IsAssignableFrom<ITokenCountingService>(service);
    }

    private static UnifiedRequest AdvancedRequest(string model) => new()
    {
        Model = model,
        Instructions = "Be concise.",
        Messages = { new UnifiedMessage(MessageRole.User, "Weather in Boston?") },
        Tools = new List<ToolDefinition>
        {
            new()
            {
                Name = "get_weather",
                Description = "Get weather.",
                Parameters = new Dictionary<string, object>
                {
                    ["type"] = "object",
                    ["properties"] = new Dictionary<string, object>
                    {
                        ["city"] = new Dictionary<string, object> { ["type"] = "string" }
                    },
                    ["required"] = new[] { "city" },
                    ["additionalProperties"] = false
                }
            }
        },
        ToolChoice = new ToolChoice { Type = ToolChoiceType.Auto },
        Output = new OutputFormat
        {
            Kind = OutputFormatKind.JsonSchema,
            JsonSchema = new JsonSchemaDefinition
            {
                Name = "weather",
                Schema = new Dictionary<string, object>
                {
                    ["type"] = "object",
                    ["properties"] = new Dictionary<string, object>
                    {
                        ["temperature"] = new Dictionary<string, object> { ["type"] = "number" }
                    },
                    ["required"] = new[] { "temperature" },
                    ["additionalProperties"] = false
                }
            }
        }
    };
}
