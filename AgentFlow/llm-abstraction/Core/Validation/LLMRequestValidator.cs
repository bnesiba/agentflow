using System;
using System.Collections.Generic;
using System.Linq;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Core.Validation
{
    public enum RequestDiagnosticSeverity
    {
        Information,
        Warning,
        Error
    }

    public sealed class RequestDiagnostic
    {
        public RequestDiagnosticSeverity Severity { get; init; }
        public string Code { get; init; } = string.Empty;
        public string Message { get; init; } = string.Empty;
        public string? PropertyPath { get; init; }
    }

    public sealed class LLMRequestValidationException : ArgumentException
    {
        public LLMRequestValidationException(
            LLMProvider provider,
            IReadOnlyList<RequestDiagnostic> diagnostics)
            : base(string.Join(Environment.NewLine, diagnostics.Select(diagnostic => diagnostic.Message)))
        {
            Provider = provider;
            Diagnostics = diagnostics;
        }

        public LLMProvider Provider { get; }
        public IReadOnlyList<RequestDiagnostic> Diagnostics { get; }
    }

    public sealed class ProviderCapabilityProfile
    {
        public required LLMProvider Provider { get; init; }
        public bool SupportsStreaming { get; init; }
        public bool SupportsCustomTools { get; init; }
        public bool SupportsParallelToolCalls { get; init; }
        public bool SupportsImages { get; init; }
        public bool SupportsDocuments { get; init; }
        public bool SupportsStructuredOutput { get; init; }
        public bool SupportsReasoning { get; init; }
        public double MinimumTemperature { get; init; }
        public double MaximumTemperature { get; init; }
    }

    public static class LLMRequestValidator
    {
        public static ProviderCapabilityProfile GetCapabilities(LLMProvider provider)
        {
            return provider switch
            {
                LLMProvider.OpenAI => new ProviderCapabilityProfile
                {
                    Provider = provider,
                    SupportsStreaming = true,
                    SupportsCustomTools = true,
                    SupportsParallelToolCalls = true,
                    SupportsImages = true,
                    SupportsDocuments = true,
                    SupportsStructuredOutput = true,
                    SupportsReasoning = true,
                    MinimumTemperature = 0,
                    MaximumTemperature = 2
                },
                LLMProvider.Claude => new ProviderCapabilityProfile
                {
                    Provider = provider,
                    SupportsStreaming = true,
                    SupportsCustomTools = true,
                    SupportsParallelToolCalls = true,
                    SupportsImages = true,
                    SupportsDocuments = true,
                    SupportsStructuredOutput = true,
                    SupportsReasoning = true,
                    MinimumTemperature = 0,
                    MaximumTemperature = 1
                },
                LLMProvider.Gemini => new ProviderCapabilityProfile
                {
                    Provider = provider,
                    SupportsStreaming = true,
                    SupportsCustomTools = true,
                    SupportsParallelToolCalls = true,
                    SupportsImages = true,
                    SupportsDocuments = true,
                    SupportsStructuredOutput = true,
                    SupportsReasoning = true,
                    MinimumTemperature = 0,
                    MaximumTemperature = 2
                },
                _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, null)
            };
        }

        public static IReadOnlyList<RequestDiagnostic> Validate(
            UnifiedRequest request,
            LLMProvider provider)
        {
            ArgumentNullException.ThrowIfNull(request);
            var diagnostics = new List<RequestDiagnostic>();
            var capabilities = GetCapabilities(provider);

            AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(request.Model),
                "request.model.required", "A provider model identifier is required.", "Model");
            AddErrorIf(diagnostics,
                !request.Messages.Any(message => message.Role != MessageRole.System),
                "request.messages.required",
                "At least one non-system conversation message is required.",
                "Messages");
            AddErrorIf(diagnostics, request.Parameters.MaxOutputTokens < 0,
                "request.max_output_tokens.range",
                "MaxOutputTokens cannot be negative.",
                "Parameters.MaxOutputTokens");

            if (request.Parameters.Temperature is double temperature)
            {
                AddErrorIf(diagnostics,
                    temperature < capabilities.MinimumTemperature || temperature > capabilities.MaximumTemperature,
                    "request.temperature.range",
                    $"{provider} temperature must be between {capabilities.MinimumTemperature} and {capabilities.MaximumTemperature}.",
                    "Parameters.Temperature");
            }

            if (request.Parameters.TopP is double topP)
            {
                AddErrorIf(diagnostics, topP < 0 || topP > 1,
                    "request.top_p.range", "TopP must be between 0 and 1.", "Parameters.TopP");
            }

            AddErrorIf(diagnostics, request.Parameters.TopK <= 0,
                "request.top_k.range", "TopK must be greater than zero.", "Parameters.TopK");

            ValidateTools(request, diagnostics);
            ValidateResponseFormat(request, diagnostics);
            ValidateProviderRules(request, provider, diagnostics);

            if (request.Continuation != null && request.Continuation.Provider != ProviderName(provider))
            {
                diagnostics.Add(new RequestDiagnostic
                {
                    Severity = RequestDiagnosticSeverity.Warning,
                    Code = "request.continuation.provider_mismatch",
                    Message = $"Continuation state for '{request.Continuation.Provider}' will not be sent to {provider}.",
                    PropertyPath = "Continuation.Provider"
                });
            }

            return diagnostics;
        }

        public static void ValidateAndThrow(UnifiedRequest request, LLMProvider provider)
        {
            var errors = Validate(request, provider)
                .Where(diagnostic => diagnostic.Severity == RequestDiagnosticSeverity.Error)
                .ToList();
            if (errors.Count > 0)
                throw new LLMRequestValidationException(provider, errors);
        }

        private static void ValidateTools(
            UnifiedRequest request,
            List<RequestDiagnostic> diagnostics)
        {
            var tools = request.Tools ?? new List<ToolDefinition>();
            for (var index = 0; index < tools.Count; index++)
            {
                AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(tools[index].Name),
                    "request.tool.name_required", "Every tool requires a name.", $"Tools[{index}].Name");
                AddErrorIf(diagnostics, tools[index].Parameters.Count == 0,
                    "request.tool.schema_required", $"Tool '{tools[index].Name}' requires a JSON input schema.", $"Tools[{index}].Parameters");
            }

            foreach (var duplicate in tools
                .Where(tool => !string.IsNullOrWhiteSpace(tool.Name))
                .GroupBy(tool => tool.Name, StringComparer.Ordinal)
                .Where(group => group.Count() > 1))
            {
                diagnostics.Add(Error(
                    "request.tool.name_duplicate",
                    $"Tool name '{duplicate.Key}' is defined more than once.",
                    "Tools"));
            }

            if (request.ToolChoice?.Type == ToolChoiceType.Specific)
            {
                AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(request.ToolChoice.ToolName),
                    "request.tool_choice.name_required",
                    "Specific tool choice requires ToolName.",
                    "ToolChoice.ToolName");
                AddErrorIf(diagnostics,
                    !string.IsNullOrWhiteSpace(request.ToolChoice.ToolName) &&
                    tools.All(tool => tool.Name != request.ToolChoice.ToolName),
                    "request.tool_choice.name_unknown",
                    $"Specific tool '{request.ToolChoice.ToolName}' is not present in Tools.",
                    "ToolChoice.ToolName");
            }
        }

        private static void ValidateResponseFormat(
            UnifiedRequest request,
            List<RequestDiagnostic> diagnostics)
        {
            if (request.ResponseFormat?.Type != ResponseFormatType.JsonSchema)
                return;

            AddErrorIf(diagnostics, request.ResponseFormat.JsonSchema == null,
                "request.response_schema.required",
                "JsonSchema response format requires a schema definition.",
                "ResponseFormat.JsonSchema");
            if (request.ResponseFormat.JsonSchema != null)
            {
                AddErrorIf(diagnostics, request.ResponseFormat.JsonSchema.Schema.Count == 0,
                    "request.response_schema.empty",
                    "Response JSON Schema cannot be empty.",
                    "ResponseFormat.JsonSchema.Schema");
            }
        }

        private static void ValidateProviderRules(
            UnifiedRequest request,
            LLMProvider provider,
            List<RequestDiagnostic> diagnostics)
        {
            if (provider == LLMProvider.OpenAI)
            {
                AddWarningIf(diagnostics, request.Parameters.TopK != null,
                    "openai.top_k.unsupported",
                    "OpenAI Responses does not expose the portable TopK setting; it will not be sent.",
                    "Parameters.TopK");
                AddWarningIf(diagnostics, request.Parameters.StopSequences?.Count > 0,
                    "openai.stop_sequences.unsupported",
                    "OpenAI Responses does not expose portable stop sequences; they will not be sent.",
                    "Parameters.StopSequences");
                AddWarningIf(diagnostics, request.Reasoning?.BudgetTokens != null,
                    "openai.reasoning_budget.unsupported",
                    "OpenAI reasoning does not use the portable token budget; it will not be sent.",
                    "Reasoning.BudgetTokens");
            }

            if (provider == LLMProvider.Claude &&
                request.Reasoning?.Enabled != false &&
                (request.Reasoning?.BudgetTokens != null ||
                 string.Equals(request.Reasoning?.Effort, "adaptive", StringComparison.OrdinalIgnoreCase)) &&
                request.ToolChoice?.Type is ToolChoiceType.Required or ToolChoiceType.Specific)
            {
                diagnostics.Add(Error(
                    "claude.thinking.forced_tool_choice",
                    "Claude thinking supports only Auto or None tool choice; forced tool choice would be rejected.",
                    "ToolChoice.Type"));
            }

            if (provider == LLMProvider.Claude && request.Reasoning?.BudgetTokens is int budget)
            {
                AddErrorIf(diagnostics, budget < 1024,
                    "claude.thinking.budget_minimum",
                    "Claude manual thinking budget must be at least 1024 tokens.",
                    "Reasoning.BudgetTokens");
                AddErrorIf(diagnostics,
                    request.Parameters.MaxOutputTokens is int maximum && budget >= maximum,
                    "claude.thinking.budget_output_limit",
                    "Claude thinking budget must be lower than MaxOutputTokens.",
                    "Reasoning.BudgetTokens");
            }

            if (provider == LLMProvider.Claude)
            {
                AddWarningIf(diagnostics, request.ToolChoice?.DisableParallelToolUse != null,
                    "claude.parallel_tool_control.unmapped",
                    "Claude DisableParallelToolUse is not mapped by the portable tool-choice converter.",
                    "ToolChoice.DisableParallelToolUse");
                AddWarningIf(diagnostics,
                    request.Reasoning?.Summary != null || request.Reasoning?.IncludeThoughts != null,
                    "claude.reasoning_display.unmapped",
                    "Claude reasoning summary/display controls are provider-specific and are not mapped by this portable request.",
                    "Reasoning");
                AddWarningIf(diagnostics, request.Metadata?.Tags?.Count > 0,
                    "claude.metadata_tags.unsupported",
                    "Claude request metadata tags are not sent; only UserId is mapped.",
                    "Metadata.Tags");
            }

            if (provider == LLMProvider.Gemini &&
                request.Reasoning?.BudgetTokens != null &&
                !string.IsNullOrWhiteSpace(request.Reasoning.Effort))
            {
                diagnostics.Add(Error(
                    "gemini.thinking.conflicting_controls",
                    "Gemini ThinkingBudget and ThinkingLevel cannot be selected together in one portable request.",
                    "Reasoning"));
            }

            if (provider == LLMProvider.Gemini)
            {
                AddWarningIf(diagnostics, request.ToolChoice?.DisableParallelToolUse != null,
                    "gemini.parallel_tool_control.unmapped",
                    "Gemini DisableParallelToolUse is not mapped by the portable tool-choice converter.",
                    "ToolChoice.DisableParallelToolUse");
                AddWarningIf(diagnostics, request.Tools?.Any(tool => tool.Strict != null) == true,
                    "gemini.tool_strict.unmapped",
                    "Gemini tool Strict values are not represented by this generateContent function declaration.",
                    "Tools.Strict");
                AddWarningIf(diagnostics, request.Reasoning?.Summary != null,
                    "gemini.reasoning_summary.unmapped",
                    "Gemini does not use the portable reasoning Summary field; it will not be sent.",
                    "Reasoning.Summary");
                AddWarningIf(diagnostics,
                    request.Metadata?.UserId != null || request.Metadata?.Tags?.Count > 0,
                    "gemini.metadata.unsupported",
                    "Portable request metadata is not sent by the Gemini generateContent converter.",
                    "Metadata");
            }

            foreach (var message in request.Messages)
            {
                foreach (var result in message.Content.OfType<ToolResultContent>())
                {
                    AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(result.ToolCallId),
                        "request.tool_result.call_id_required",
                        "Every tool result requires ToolCallId.",
                        "Messages.Content.ToolCallId");
                }
            }
        }

        private static string ProviderName(LLMProvider provider) => provider switch
        {
            LLMProvider.OpenAI => ProviderIds.OpenAI,
            LLMProvider.Claude => ProviderIds.Claude,
            LLMProvider.Gemini => ProviderIds.Gemini,
            _ => string.Empty
        };

        private static void AddErrorIf(
            List<RequestDiagnostic> diagnostics,
            bool condition,
            string code,
            string message,
            string? propertyPath)
        {
            if (condition)
                diagnostics.Add(Error(code, message, propertyPath));
        }

        private static void AddWarningIf(
            List<RequestDiagnostic> diagnostics,
            bool condition,
            string code,
            string message,
            string? propertyPath)
        {
            if (!condition)
                return;
            diagnostics.Add(new RequestDiagnostic
            {
                Severity = RequestDiagnosticSeverity.Warning,
                Code = code,
                Message = message,
                PropertyPath = propertyPath
            });
        }

        private static RequestDiagnostic Error(string code, string message, string? propertyPath)
        {
            return new RequestDiagnostic
            {
                Severity = RequestDiagnosticSeverity.Error,
                Code = code,
                Message = message,
                PropertyPath = propertyPath
            };
        }
    }
}
