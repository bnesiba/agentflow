using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class OpenAISamplingTests
{
    [Theory]
    [InlineData("gpt-5.6", ReasoningEffort.None)]
    [InlineData("gpt-5.6-terra", ReasoningEffort.Medium)]
    public void VerifiedSamplingValuesAreNeverSilentlyRemoved(
        string model,
        ReasoningEffort effort)
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
            Reasoning = new ReasoningOptions { Effort = effort }
        };

        var converted = new OpenAIConverter().ConvertRequest(request);

        Assert.Equal(0.4, converted.Temperature);
        Assert.Equal(0.8, converted.TopP);
    }

    [Fact]
    public void UnknownModelWarnsAndStillSerializesUnverifiedAdvancedControls()
    {
        var request = new UnifiedRequest
        {
            Model = "future-model-name",
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") },
            Parameters = new GenerationParameters { Temperature = 0.4 },
            Reasoning = new ReasoningOptions { Effort = ReasoningEffort.High }
        };

        var diagnostics = LLMRequestValidator.Validate(request, LLMAbstraction.Core.LLMProvider.OpenAI);
        var converted = new OpenAIConverter().ConvertRequest(request);

        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.sampling.unknown_model" &&
                          diagnostic.Severity == RequestDiagnosticSeverity.Warning);
        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.reasoning.unknown_model" &&
                          diagnostic.Severity == RequestDiagnosticSeverity.Warning);
        Assert.Equal(0.4, converted.Temperature);
        Assert.Equal("high", converted.Reasoning!.Effort);
    }
}
