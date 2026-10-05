using System.Collections.Generic;
using LLMAbstraction.Core.Transport;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// Unified request model for LLM API calls
    /// </summary>
    public class UnifiedRequest
    {
        /// <summary>
        /// Model identifier (provider-specific)
        /// </summary>
        public string Model { get; set; } = string.Empty;

        /// <summary>
        /// Conversation messages
        /// </summary>
        public List<UnifiedMessage> Messages { get; set; } = new();

        /// <summary>
        /// Developer/system-level instructions for the model.
        /// </summary>
        public string? Instructions { get; set; }

        /// <summary>
        /// Generation parameters
        /// </summary>
        public GenerationParameters Parameters { get; set; } = new();

        /// <summary>
        /// Available application functions and provider-native tools.
        /// </summary>
        public ToolCollection? Tools { get; set; }

    /// <summary>
        /// Tool choice strategy
        /// </summary>
        public ToolChoice? ToolChoice { get; set; }

        /// <summary>
        /// Requested output representation.
        /// </summary>
        public OutputFormat? Output { get; set; }

    /// <summary>
        /// Reasoning/thinking controls for models that support them.
        /// </summary>
        public ReasoningOptions? Reasoning { get; set; }

    /// <summary>
        /// Request metadata shared by providers when supported.
        /// </summary>
        public RequestMetadata? Metadata { get; set; }

    /// <summary>
        /// Provider-specific options (escape hatch)
        /// </summary>
        public ProviderOptions? ProviderOptions { get; set; }

        /// <summary>
        /// Provider-bound state from an earlier response. This is used for APIs
        /// whose continuation items cannot be represented as chat messages.
        /// </summary>
        public ProviderContinuationState? Continuation { get; set; }

        /// <summary>
        /// Optional retry override and token estimate for admission control on
        /// this request. This is never serialized to a provider payload.
        /// </summary>
        public RequestTransportOptions? Transport { get; set; }

        /// <summary>Request-wide prompt-cache intent and typed provider controls.</summary>
        public PromptCacheOptions? Cache { get; set; }

        /// <summary>Optional cache breakpoint after the normalized instructions.</summary>
        public PromptCacheDirective? InstructionsCache { get; set; }

    }

    /// <summary>
    /// Generation parameters common across providers
    /// </summary>
    public class GenerationParameters
    {
        /// <summary>
        /// Maximum tokens to include in the model output.
        /// </summary>
        public int? MaxOutputTokens { get; set; }

        /// <summary>
        /// Sampling temperature (0.0 to 2.0, provider-dependent)
        /// </summary>
        public double? Temperature { get; set; }

        /// <summary>
        /// Nucleus sampling parameter (0.0 to 1.0)
        /// </summary>
        public double? TopP { get; set; }

        /// <summary>
        /// Top-k sampling parameter
        /// </summary>
        public int? TopK { get; set; }

        /// <summary>
        /// Stop sequences
        /// </summary>
        public List<string>? StopSequences { get; set; }

        /// <summary>
        /// Request provider streaming. StreamAsync enables this automatically;
        /// callers normally leave it false when using GenerateAsync.
        /// </summary>
        public bool Stream { get; set; } = false;
    }

    /// <summary>
    /// Tool choice strategy
    /// </summary>
    public class ToolChoice
    {
        public ToolChoiceType Type { get; set; }
        public string? ToolName { get; set; }  // For specific tool selection
        public bool? DisableParallelToolUse { get; set; }
    }

    public enum ToolChoiceType
    {
        Auto,      // Model decides
        None,      // Don't use tools
        Required,  // Must use a tool
        Specific   // Use specific tool (ToolName must be set)
    }

    /// <summary>
    /// Output representation requested from the model.
    /// </summary>
    public sealed class OutputFormat
    {
        public OutputFormatKind Kind { get; set; }

        /// <summary>
        /// JSON Schema for structured output when <see cref="Kind"/> is
        /// <see cref="OutputFormatKind.JsonSchema"/>.
        /// </summary>
        public JsonSchemaDefinition? JsonSchema { get; set; }
    }

    public enum OutputFormatKind
    {
        Text,
        JsonObject,
        JsonSchema
    }

    /// <summary>
    /// JSON Schema definition for structured output
    /// </summary>
    public sealed class JsonSchemaDefinition
    {
        /// <summary>
        /// Name of the schema
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Description of the schema
        /// </summary>
        public string? Description { get; set; }

        /// <summary>
        /// JSON Schema definition (following JSON Schema spec)
        /// </summary>
        public Dictionary<string, object> Schema { get; set; } = new();

    }

    /// <summary>
    /// Portable reasoning controls with typed provider-specific extensions.
    /// Effort controls how much work the model performs; it does not select an
    /// Anthropic thinking mode or a Gemini token budget.
    /// </summary>
    public sealed class ReasoningOptions
    {
        public ReasoningEffort? Effort { get; set; }
        public ReasoningOutput? Output { get; set; }
        public AnthropicReasoningOptions? Anthropic { get; set; }
        public OpenAIReasoningOptions? OpenAI { get; set; }
        public GeminiReasoningOptions? Gemini { get; set; }
    }

    public enum ReasoningEffort
    {
        None,
        Minimal,
        Low,
        Medium,
        High,
        XHigh,
        Max
    }

    public enum ReasoningOutput
    {
        Omitted,
        Summary
    }

    public sealed class AnthropicReasoningOptions
    {
        public AnthropicThinkingMode Mode { get; set; } = AnthropicThinkingMode.Default;
        public int? BudgetTokens { get; set; }
    }

    public enum AnthropicThinkingMode
    {
        Default,
        Disabled,
        Adaptive,
        Manual
    }

    public sealed class OpenAIReasoningOptions
    {
        public OpenAIReasoningMode? Mode { get; set; }
        public OpenAIReasoningContext? Context { get; set; }
        public OpenAIReasoningSummary? Summary { get; set; }
    }

    public enum OpenAIReasoningMode
    {
        Standard,
        Pro
    }

    public enum OpenAIReasoningContext
    {
        Auto,
        CurrentTurn,
        AllTurns
    }

    public enum OpenAIReasoningSummary
    {
        Auto,
        Concise,
        Detailed
    }

    public sealed class GeminiReasoningOptions
    {
        /// <summary>
        /// Legacy generateContent token budget. Interactions uses
        /// <see cref="ReasoningOptions.Effort"/> as a thinking level instead.
        /// </summary>
        public int? ThinkingBudget { get; set; }
    }

    /// <summary>
    /// Request-level metadata.
    /// </summary>
    public class RequestMetadata
    {
        public string? UserId { get; set; }
        public Dictionary<string, string>? Tags { get; set; }
    }

    /// <summary>
    /// Provider-specific options (escape hatch for unique features)
    /// </summary>
    public class ProviderOptions
    {
        public Dictionary<string, object>? OpenAI { get; set; }
        public Dictionary<string, object>? Gemini { get; set; }
        public Dictionary<string, object>? Claude { get; set; }
    }
}
