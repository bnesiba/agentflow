using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class OpenAISamplingTests
{
    [Theory]
    [InlineData("gpt-5", null)]
    [InlineData("gpt-5.6", "none")]
    [InlineData("gpt-5.6-terra", "medium")]
    [InlineData("gpt-4o", null)]
    [InlineData("future-model-name", "future-effort")]
    public void CallerSuppliedSamplingValuesAreNeverSilentlyRemoved(
        string model,
        string? effort)
    {
        var request = new UnifiedRequest
        {
            Model = model,
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") },
            Parameters = new GenerationParameters
            {
                Temperature = 0.4,
                TopP = 0.8
            },
            Reasoning = effort == null ? null : new ReasoningOptions { Effort = effort }
        };

        var converted = new OpenAIConverter().ConvertRequest(request);

        Assert.Equal(0.4, converted.Temperature);
        Assert.Equal(0.8, converted.TopP);
    }
}
