using System.Text.Json;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Claude.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.Gemini.Models;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.OpenAI.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class ResponseMetadataTests
{
    [Fact]
    public void OpenAIRefusalAndAnnotationsAreNotFlattenedOrDiscarded()
    {
        const string json = """
        {
          "id":"resp_1","model":"gpt-5.6","status":"incomplete",
          "incomplete_details":{"reason":"content_filter"},
          "output":[{"id":"msg_1","type":"message","role":"assistant","content":[
            {"type":"output_text","text":"Partial","annotations":[{"type":"url_citation","url":"https://example.com"}]},
            {"type":"refusal","refusal":"I cannot help with that."}
          ]}]
        }
        """;
        var native = JsonSerializer.Deserialize<OpenAIResponse>(json)!;

        var response = new OpenAIConverter().ConvertResponse(native);
        var choice = Assert.Single(response.Choices);

        Assert.Equal(FinishReason.ContentFilter, choice.FinishReason);
        var text = Assert.IsType<TextContent>(choice.Message.Content[0]);
        Assert.True(text.ProviderMetadata!.ContainsKey("openai.annotations"));
        Assert.Equal("I cannot help with that.",
            Assert.IsType<RefusalContent>(choice.Message.Content[1]).Refusal);
        Assert.Equal("content_filter", response.ProviderMetadata!["incompleteReason"]);
    }

    [Fact]
    public void GeminiSafetyAndNativeFinishReasonArePreservedPerChoice()
    {
        var native = new GeminiGenerateResponse
        {
            Candidates =
            {
                new GeminiCandidate
                {
                    Index = 4,
                    FinishReason = "SAFETY",
                    Content = new GeminiContent(),
                    SafetyRatings = new List<GeminiSafetyRating>
                    {
                        new GeminiSafetyRating
                        {
                            Category = "HARM_CATEGORY_DANGEROUS_CONTENT",
                            Probability = "HIGH"
                        }
                    }
                }
            }
        };

        var choice = Assert.Single(new GeminiConverter().ConvertResponse(native).Choices);

        Assert.Equal(4, choice.Index);
        Assert.Equal(FinishReason.ContentFilter, choice.FinishReason);
        Assert.Equal("SAFETY", choice.ProviderMetadata!["gemini.finishReason"]);
        Assert.Single(Assert.IsType<List<GeminiSafetyRating>>(
            choice.ProviderMetadata["gemini.safetyRatings"]));
    }

    [Fact]
    public void ClaudeStopAndDetailedCacheMetadataArePreserved()
    {
        var native = new ClaudeMessageResponse
        {
            Id = "msg_1",
            Model = "claude-sonnet-4-6",
            StopReason = "stop_sequence",
            StopSequence = "END",
            Usage = new ClaudeUsage
            {
                InputTokens = 10,
                OutputTokens = 2,
                Ephemeral5mInputTokens = 3,
                Ephemeral1hInputTokens = 4
            }
        };

        var response = new ClaudeConverter().ConvertResponse(native);

        Assert.Equal("stop_sequence", response.ProviderMetadata!["claude.stopReason"]);
        Assert.Equal("END", response.ProviderMetadata["claude.stopSequence"]);
        Assert.Equal(3, response.ProviderMetadata["claude.cacheCreationEphemeral5mTokens"]);
        Assert.Equal(4, response.ProviderMetadata["claude.cacheCreationEphemeral1hTokens"]);
    }

    [Theory]
    [InlineData("BLOCKLIST", FinishReason.ContentFilter)]
    [InlineData("PROHIBITED_CONTENT", FinishReason.ContentFilter)]
    [InlineData("MALFORMED_FUNCTION_CALL", FinishReason.Error)]
    [InlineData("UNEXPECTED_TOOL_CALL", FinishReason.Error)]
    public void GeminiModernFinishReasonsAreNormalized(string nativeReason, FinishReason expected)
    {
        var response = new GeminiGenerateResponse
        {
            Candidates =
            {
                new GeminiCandidate
                {
                    FinishReason = nativeReason,
                    Content = new GeminiContent()
                }
            }
        };

        Assert.Equal(expected,
            Assert.Single(new GeminiConverter().ConvertResponse(response).Choices).FinishReason);
    }
}
