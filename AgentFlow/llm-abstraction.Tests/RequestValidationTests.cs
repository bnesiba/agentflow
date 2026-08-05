using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Providers.Claude;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class RequestValidationTests
{
    [Fact]
    public void CommonValidationReturnsStructuredDiagnostics()
    {
        var request = new UnifiedRequest
        {
            Model = "",
            Parameters = new GenerationParameters
            {
                MaxOutputTokens = -1,
                Temperature = 3,
                TopP = -0.1,
                TopK = 0
            }
        };

        var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.OpenAI);

        Assert.Contains(diagnostics, diagnostic => diagnostic.Code == "request.model.required");
        Assert.Contains(diagnostics, diagnostic => diagnostic.Code == "request.messages.required");
        Assert.Contains(diagnostics, diagnostic => diagnostic.Code == "request.max_output_tokens.range");
        Assert.Contains(diagnostics, diagnostic => diagnostic.Code == "request.temperature.range");
        Assert.Contains(diagnostics, diagnostic => diagnostic.Code == "request.top_p.range");
        Assert.Contains(diagnostics, diagnostic => diagnostic.Code == "request.top_k.range");
        Assert.All(diagnostics, diagnostic => Assert.False(string.IsNullOrWhiteSpace(diagnostic.PropertyPath)));
    }

    [Fact]
    public void ConverterThrowsValidationExceptionWithProviderAndCodes()
    {
        var request = new UnifiedRequest
        {
            Model = "claude-sonnet-4-6",
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") },
            Parameters = new GenerationParameters { MaxOutputTokens = 2000 },
            Tools = new List<ToolDefinition>
            {
                new ToolDefinition
                {
                    Name = "weather",
                    Parameters = new Dictionary<string, object> { ["type"] = "object" }
                }
            },
            ToolChoice = new ToolChoice { Type = ToolChoiceType.Required },
            Reasoning = new ReasoningOptions { BudgetTokens = 1024 }
        };

        var exception = Assert.Throws<LLMRequestValidationException>(
            () => new ClaudeConverter().ConvertRequest(request));

        Assert.Equal(LLMProvider.Claude, exception.Provider);
        Assert.Contains(exception.Diagnostics,
            diagnostic => diagnostic.Code == "claude.thinking.forced_tool_choice");
    }

    [Fact]
    public void SpecificToolChoiceMustNameAnExistingTool()
    {
        var request = new UnifiedRequest
        {
            Model = "gpt-5.6",
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") },
            Tools = new List<ToolDefinition>
            {
                new ToolDefinition
                {
                    Name = "known_tool",
                    Parameters = new Dictionary<string, object> { ["type"] = "object" }
                }
            },
            ToolChoice = new ToolChoice
            {
                Type = ToolChoiceType.Specific,
                ToolName = "unknown_tool"
            }
        };

        var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.OpenAI);
        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.tool_choice.name_unknown");
    }

    [Fact]
    public void ProviderMismatchProducesWarningRatherThanSilentUnknownBehavior()
    {
        var request = new UnifiedRequest
        {
            Model = "gpt-5.6",
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") },
            Continuation = new ProviderContinuationState { Provider = ProviderIds.Claude }
        };

        var warning = Assert.Single(
            LLMRequestValidator.Validate(request, LLMProvider.OpenAI),
            diagnostic => diagnostic.Code == "request.continuation.provider_mismatch");
        Assert.Equal(RequestDiagnosticSeverity.Warning, warning.Severity);
    }

    [Fact]
    public void CapabilityProfilesExposePortableFeatureBounds()
    {
        var claude = LLMRequestValidator.GetCapabilities(LLMProvider.Claude);
        var openAI = LLMRequestValidator.GetCapabilities(LLMProvider.OpenAI);

        Assert.Equal(1, claude.MaximumTemperature);
        Assert.Equal(2, openAI.MaximumTemperature);
        Assert.True(claude.SupportsReasoning);
        Assert.True(openAI.SupportsStructuredOutput);
    }

    [Fact]
    public void LossyProviderMappingsAreReportedAsWarnings()
    {
        var openAIRequest = new UnifiedRequest
        {
            Model = "gpt-5.6",
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") },
            Parameters = new GenerationParameters
            {
                TopK = 10,
                StopSequences = new List<string> { "END" }
            }
        };
        var openAIWarnings = LLMRequestValidator.Validate(openAIRequest, LLMProvider.OpenAI);
        Assert.Contains(openAIWarnings, diagnostic => diagnostic.Code == "openai.top_k.unsupported");
        Assert.Contains(openAIWarnings, diagnostic => diagnostic.Code == "openai.stop_sequences.unsupported");

        var geminiRequest = new UnifiedRequest
        {
            Model = "gemini-3.5-flash",
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") },
            Tools = new List<ToolDefinition>
            {
                new()
                {
                    Name = "tool",
                    Strict = true,
                    Parameters = new Dictionary<string, object> { ["type"] = "object" }
                }
            }
        };
        var geminiWarning = Assert.Single(
            LLMRequestValidator.Validate(geminiRequest, LLMProvider.Gemini),
            diagnostic => diagnostic.Code == "gemini.tool_strict.unmapped");
        Assert.Equal(RequestDiagnosticSeverity.Warning, geminiWarning.Severity);
    }
}
