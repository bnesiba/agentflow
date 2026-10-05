using System.Text.Json;
using System.Text.Json.Nodes;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class ProviderOptionsTests
{
    [Fact]
    public void OpenAIAdditionalOptionsAreSerializedAtRequestRoot()
    {
        var request = BasicRequest("gpt-5.6");
        request.ProviderOptions = new ProviderOptions
        {
            OpenAI = new Dictionary<string, object>
            {
                ["store"] = false,
                ["service_tier"] = "flex",
                ["include"] = new[] { "reasoning.encrypted_content" }
            }
        };

        var json = Serialize(new OpenAIConverter().ConvertRequest(request));

        Assert.False(json["store"]!.GetValue<bool>());
        Assert.Equal("flex", json["service_tier"]!.GetValue<string>());
        Assert.Equal("reasoning.encrypted_content", json["include"]![0]!.GetValue<string>());
    }

    [Fact]
    public void ClaudeAdditionalOptionsAreSerializedButHeaderOptionIsNotInBody()
    {
        var request = BasicRequest("claude-sonnet-4-6");
        request.ProviderOptions = new ProviderOptions
        {
            Claude = new Dictionary<string, object>
            {
                ["anthropicBeta"] = "files-api-2025-04-14",
                ["service_tier"] = "auto",
                ["context_management"] = new Dictionary<string, object>
                {
                    ["edits"] = Array.Empty<object>()
                }
            }
        };

        var json = Serialize(new ClaudeConverter().ConvertRequest(request));

        Assert.Null(json["anthropicBeta"]);
        Assert.Equal("auto", json["service_tier"]!.GetValue<string>());
        Assert.NotNull(json["context_management"]);
    }

    [Fact]
    public void GeminiAdditionalOptionsAndSpecialSafetySettingsAreSerialized()
    {
        var request = BasicRequest("gemini-3.5-flash");
        request.Cache = new PromptCacheOptions
        {
            Gemini = new GeminiPromptCacheOptions { CachedContentName = "cachedContents/123" }
        };
        request.ProviderOptions = new ProviderOptions
        {
            Gemini = new Dictionary<string, object>
            {
                ["safetySettings"] = new List<Dictionary<string, object>>
                {
                    new()
                    {
                        ["category"] = "HARM_CATEGORY_HATE_SPEECH",
                        ["threshold"] = "BLOCK_ONLY_HIGH"
                    }
                }
            }
        };

        var json = Serialize(new GeminiConverter().ConvertRequest(request));

        Assert.Equal("cachedContents/123", json["cachedContent"]!.GetValue<string>());
        Assert.Equal("HARM_CATEGORY_HATE_SPEECH",
            json["safetySettings"]![0]!["category"]!.GetValue<string>());
    }

    [Theory]
    [InlineData("openai")]
    [InlineData("claude")]
    [InlineData("gemini")]
    public void AdditionalOptionsCannotOverrideUnifiedFields(string provider)
    {
        var request = BasicRequest(provider switch
        {
            "openai" => "gpt-5.6",
            "claude" => "claude-sonnet-4-6",
            _ => "gemini-3.5-flash"
        });
        request.ProviderOptions = provider switch
        {
            "openai" => new ProviderOptions
            {
                OpenAI = new Dictionary<string, object> { ["model"] = "overridden" }
            },
            "claude" => new ProviderOptions
            {
                Claude = new Dictionary<string, object> { ["messages"] = Array.Empty<object>() }
            },
            _ => new ProviderOptions
            {
                Gemini = new Dictionary<string, object> { ["generationConfig"] = new { temperature = 2 } }
            }
        };

        var error = provider switch
        {
            "openai" => Assert.Throws<ArgumentException>(() => new OpenAIConverter().ConvertRequest(request)),
            "claude" => Assert.Throws<ArgumentException>(() => new ClaudeConverter().ConvertRequest(request)),
            _ => Assert.Throws<ArgumentException>(() => new GeminiConverter().ConvertRequest(request))
        };

        Assert.Contains("cannot override", error.Message);
    }

    private static UnifiedRequest BasicRequest(string model) => new()
    {
        Model = model,
        Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
    };

    private static JsonObject Serialize(object value)
        => JsonSerializer.SerializeToNode(value)!.AsObject();
}
