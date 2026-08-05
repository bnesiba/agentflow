using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class CapabilityAndStructuredOutputTests
{
    [Fact]
    public void RegistryResolvesSnapshotsAndKeepsUnknownModelsConservative()
    {
        var snapshot = ModelCapabilityRegistry.Resolve(
            LLMProvider.Claude,
            "claude-sonnet-4-6-20260801",
            LLMApiSurface.AnthropicMessages);
        var unknown = ModelCapabilityRegistry.Resolve(
            LLMProvider.OpenAI,
            "gpt-future",
            LLMApiSurface.OpenAIResponses);

        Assert.Equal(ModelRecognition.Known, snapshot.Recognition);
        Assert.Equal("claude-4.6", snapshot.Family);
        Assert.Equal(ModelRecognition.Unknown, unknown.Recognition);
        Assert.Equal(CapabilitySupport.Unknown, unknown.StructuredOutput);
        Assert.Equal(CapabilitySupport.Unknown, unknown.Reasoning);
    }

    [Fact]
    public void RegistryRejectsProviderSurfaceMismatch()
    {
        Assert.Throws<ArgumentException>(() => ModelCapabilityRegistry.Resolve(
            LLMProvider.OpenAI,
            "gpt-5.6",
            LLMApiSurface.GeminiInteractions));
    }

    [Fact]
    public void UnknownModelStillAllowsBasicTextRequest()
    {
        var converted = new OpenAIConverter().ConvertRequest(Basic("future-model"));
        Assert.Equal("future-model", converted.Model);
    }

    [Fact]
    public void UnknownModelAdvancedControlsAreSentAndRemainVisibleAsWarnings()
    {
        var request = Basic("gpt-future");
        request.Parameters.Temperature = 0.4;
        request.Reasoning = new ReasoningOptions { Effort = ReasoningEffort.High };

        var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.OpenAI);
        var converted = new OpenAIConverter().ConvertRequest(request);

        Assert.Contains(diagnostics, diagnostic =>
            diagnostic.Code == "request.reasoning.unknown_model" &&
            diagnostic.Severity == RequestDiagnosticSeverity.Warning);
        Assert.Contains(diagnostics, diagnostic =>
            diagnostic.Code == "request.sampling.unknown_model" &&
            diagnostic.Severity == RequestDiagnosticSeverity.Warning);
        Assert.Equal(0.4, converted.Temperature);
        Assert.Equal("high", converted.Reasoning!.Effort);
    }

    [Fact]
    public void DocumentedModelOutputMismatchWarnsButStillSerializes()
    {
        var request = Basic("gpt-5.4-pro");
        request.Output = ValidOutput();

        var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.OpenAI);
        var converted = new OpenAIConverter().ConvertRequest(request);

        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.output.unsupported" &&
                          diagnostic.Severity == RequestDiagnosticSeverity.Warning);
        Assert.NotNull(converted.Text?.Format);
    }

    [Fact]
    public void AnthropicJsonObjectModeIsNotPretendedToBeSchemaOutput()
    {
        var request = Basic("claude-sonnet-4-6");
        request.Output = new OutputFormat { Kind = OutputFormatKind.JsonObject };

        var exception = Assert.Throws<LLMRequestValidationException>(
            () => new ClaudeConverter().ConvertRequest(request));

        Assert.Contains(exception.Diagnostics,
            diagnostic => diagnostic.Code == "claude.output.json_object_unsupported");
    }

    [Theory]
    [InlineData(ReasoningEffort.None, true)]
    [InlineData(ReasoningEffort.Minimal, false)]
    [InlineData(ReasoningEffort.Low, true)]
    [InlineData(ReasoningEffort.Medium, true)]
    [InlineData(ReasoningEffort.High, true)]
    [InlineData(ReasoningEffort.XHigh, true)]
    [InlineData(ReasoningEffort.Max, true)]
    public void OpenAI56EffortMatrixIsTypedAndValidated(ReasoningEffort effort, bool supported)
    {
        var request = Basic("gpt-5.6");
        request.Reasoning = new ReasoningOptions { Effort = effort };

        var diagnostics = LLMRequestValidator.Validate(
            request,
            LLMProvider.OpenAI,
            LLMApiSurface.OpenAIResponses);

        Assert.Equal(!supported, diagnostics.Any(diagnostic =>
            diagnostic.Code == "request.reasoning.effort_unsupported"));
    }

    [Fact]
    public void AnthropicModeAndEffortAreIndependentAndModelAware()
    {
        var request = Basic("claude-sonnet-5");
        request.Parameters.MaxOutputTokens = 4096;
        request.Reasoning = new ReasoningOptions
        {
            Effort = ReasoningEffort.High,
            Anthropic = new AnthropicReasoningOptions
            {
                Mode = AnthropicThinkingMode.Manual,
                BudgetTokens = 1024
            }
        };

        var diagnostics = LLMRequestValidator.Validate(
            request,
            LLMProvider.Claude,
            LLMApiSurface.AnthropicMessages);

        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "claude.thinking.manual_unsupported");
        Assert.DoesNotContain(diagnostics,
            diagnostic => diagnostic.Code == "request.reasoning.effort_unsupported");

        var converted = new ClaudeConverter().ConvertRequest(request);
        Assert.Equal("enabled", converted.Thinking!.Type);
        Assert.Equal(1024, converted.Thinking.BudgetTokens);
    }

    [Fact]
    public void KnownModelFeatureMismatchWarnsButNeverActsAsAnAllowlist()
    {
        var request = Basic("gpt-4o");
        request.Reasoning = new ReasoningOptions { Effort = ReasoningEffort.High };

        var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.OpenAI);
        var converted = new OpenAIConverter().ConvertRequest(request);

        Assert.Contains(diagnostics, diagnostic =>
            diagnostic.Code == "request.reasoning.unsupported" &&
            diagnostic.Severity == RequestDiagnosticSeverity.Warning);
        Assert.Equal("high", converted.Reasoning!.Effort);
    }

    [Fact]
    public void ModelSpecificSamplingWarningDoesNotRemoveSamplingValues()
    {
        var request = Basic("claude-sonnet-5");
        request.Parameters.Temperature = 0.25;

        var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.Claude);
        var converted = new ClaudeConverter().ConvertRequest(request);

        Assert.Contains(diagnostics, diagnostic =>
            diagnostic.Code == "request.sampling.unsupported" &&
            diagnostic.Severity == RequestDiagnosticSeverity.Warning);
        Assert.Equal(0.25, converted.Temperature);
    }

    [Fact]
    public void AnthropicThinkingRejectsSamplingThatConverterWouldOtherwiseDrop()
    {
        var request = Basic("claude-sonnet-4-6");
        request.Parameters.MaxOutputTokens = 4096;
        request.Parameters.Temperature = 0.5;
        request.Parameters.TopK = 20;
        request.Reasoning = new ReasoningOptions
        {
            Anthropic = new AnthropicReasoningOptions
            {
                Mode = AnthropicThinkingMode.Manual,
                BudgetTokens = 1024
            }
        };

        var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.Claude);

        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "claude.thinking.temperature_unsupported");
        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "claude.thinking.top_k_unsupported");
    }

    [Fact]
    public void GeminiBudgetIsEndpointAndModelSpecific()
    {
        var request = Basic("gemini-2.5-flash");
        request.Reasoning = new ReasoningOptions
        {
            Gemini = new GeminiReasoningOptions { ThinkingBudget = 2048 }
        };

        Assert.DoesNotContain(
            LLMRequestValidator.Validate(
                request,
                LLMProvider.Gemini,
                LLMApiSurface.GeminiGenerateContent),
            diagnostic => diagnostic.Severity == RequestDiagnosticSeverity.Error);
        Assert.Contains(
            LLMRequestValidator.Validate(
                request,
                LLMProvider.Gemini,
                LLMApiSurface.GeminiInteractions),
            diagnostic => diagnostic.Code == "gemini.thinking.budget_unsupported");
    }

    [Fact]
    public void DirectGeminiInteractionsConversionRejectsUnrepresentableSampling()
    {
        var request = Basic("gemini-3.5-flash");
        request.Parameters.TopP = 0.8;

        var exception = Assert.Throws<LLMRequestValidationException>(
            () => new GeminiInteractionsConverter().ConvertRequest(request));

        Assert.Contains(exception.Diagnostics,
            diagnostic => diagnostic.Code == "gemini.interactions.top_p_unrepresentable");
    }

    [Fact]
    public void Gemini25GenerateContentWarnsAndStillSendsThinkingLevel()
    {
        var request = Basic("gemini-2.5-flash");
        request.Reasoning = new ReasoningOptions { Effort = ReasoningEffort.Medium };

        var diagnostics = LLMRequestValidator.Validate(
            request,
            LLMProvider.Gemini,
            LLMApiSurface.GeminiGenerateContent);
        var converted = new GeminiConverter().ConvertRequest(request);

        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.reasoning.effort_unsupported" &&
                          diagnostic.Severity == RequestDiagnosticSeverity.Warning);
        Assert.Equal("MEDIUM", converted.GenerationConfig!.ThinkingConfig!.ThinkingLevel);
    }

    [Fact]
    public void PortableSchemaAcceptsNestedRequiredObjectsAndNullableUnion()
    {
        var definition = new JsonSchemaDefinition
        {
            Name = "result",
            Schema = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object>
                {
                    ["status"] = new Dictionary<string, object>
                    {
                        ["type"] = "string",
                        ["enum"] = new[] { "ok", "failed" }
                    },
                    ["detail"] = new Dictionary<string, object>
                    {
                        ["type"] = new[] { "string", "null" }
                    }
                },
                ["required"] = new[] { "status", "detail" },
                ["additionalProperties"] = false
            }
        };

        Assert.Empty(JsonSchemaAnalyzer.Analyze(definition));
    }

    [Fact]
    public void PortableSchemaRejectsLossyConstraintsAndNonStrictObjects()
    {
        var definition = new JsonSchemaDefinition
        {
            Name = "result",
            Schema = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object>
                {
                    ["score"] = new Dictionary<string, object>
                    {
                        ["type"] = "number",
                        ["minimum"] = 0
                    },
                    ["note"] = new Dictionary<string, object> { ["type"] = "string" }
                },
                ["required"] = new[] { "score" }
            }
        };

        var diagnostics = JsonSchemaAnalyzer.Analyze(definition);

        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.response_schema.keyword_unsupported");
        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.response_schema.all_properties_required");
        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.response_schema.additional_properties");
    }

    [Fact]
    public void OutputValidationRequiresNameAndPortableSchemaShape()
    {
        var request = Basic("gpt-5.6");
        request.Output = new OutputFormat
        {
            Kind = OutputFormatKind.JsonSchema,
            JsonSchema = new JsonSchemaDefinition
            {
                Schema = new Dictionary<string, object>
                {
                    ["type"] = "array",
                    ["items"] = new Dictionary<string, object> { ["type"] = "string" }
                }
            }
        };

        var diagnostics = LLMRequestValidator.Validate(request, LLMProvider.OpenAI);

        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.response_schema.name_required");
        Assert.Contains(diagnostics,
            diagnostic => diagnostic.Code == "request.response_schema.root_type");
    }

    private static UnifiedRequest Basic(string model) => new()
    {
        Model = model,
        Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
    };

    private static OutputFormat ValidOutput() => new()
    {
        Kind = OutputFormatKind.JsonSchema,
        JsonSchema = new JsonSchemaDefinition
        {
            Name = "result",
            Schema = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object>
                {
                    ["answer"] = new Dictionary<string, object> { ["type"] = "string" }
                },
                ["required"] = new[] { "answer" },
                ["additionalProperties"] = false
            }
        }
    };
}
