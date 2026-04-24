using System.Collections.Generic;

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
        /// Backward-compatible alias for Instructions.
        /// </summary>
        [System.Obsolete("Use Instructions instead.")]
        public string? System
        {
            get => Instructions;
            set => Instructions = value;
        }

        /// <summary>
        /// Generation parameters
        /// </summary>
        public GenerationParameters Parameters { get; set; } = new();

        /// <summary>
        /// Available tools/functions
        /// </summary>
        public List<ToolDefinition>? Tools { get; set; }

    /// <summary>
    /// Tool choice strategy
    /// </summary>
    public ToolChoice? ToolChoice { get; set; }

    /// <summary>
    /// Structured output schema (JSON Schema)
    /// </summary>
    public ResponseFormat? ResponseFormat { get; set; }

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
        /// Backward-compatible alias for MaxOutputTokens.
        /// </summary>
        [System.Obsolete("Use MaxOutputTokens instead.")]
        public int? MaxTokens
        {
            get => MaxOutputTokens;
            set => MaxOutputTokens = value;
        }

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
        /// Enable streaming (for future implementation)
        /// </summary>
        public bool Stream { get; set; } = false;
    }

    /// <summary>
    /// Tool/function definition
    /// </summary>
    public class ToolDefinition
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public Dictionary<string, object> Parameters { get; set; } = new();  // JSON Schema
        public bool? Strict { get; set; }
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
    /// Response format for structured output
    /// </summary>
    public class ResponseFormat
    {
        /// <summary>
        /// Type of response format
        /// </summary>
        public ResponseFormatType Type { get; set; }

        /// <summary>
        /// JSON Schema for structured output (when Type is JsonSchema)
        /// </summary>
        public JsonSchema? JsonSchema { get; set; }
    }

    public enum ResponseFormatType
    {
        Text,       // Plain text response (default)
        Json,       // JSON object (not strictly validated)
        JsonSchema  // Structured output with schema validation
    }

    /// <summary>
    /// JSON Schema definition for structured output
    /// </summary>
    public class JsonSchema
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

        /// <summary>
        /// Whether to enforce strict schema validation
        /// </summary>
        public bool Strict { get; set; } = true;
    }

    /// <summary>
    /// Reasoning/thinking controls that map to provider-specific options.
    /// </summary>
    public class ReasoningOptions
    {
        public bool? Enabled { get; set; }
        public int? BudgetTokens { get; set; }
        public string? Effort { get; set; }
        public string? Summary { get; set; }
        public bool? IncludeThoughts { get; set; }
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
