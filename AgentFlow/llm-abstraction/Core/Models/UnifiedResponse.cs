using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// Unified response model from LLM API calls
    /// </summary>
    public class UnifiedResponse
    {
        /// <summary>
        /// Unique identifier for this response
        /// </summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>
        /// Model that generated the response
        /// </summary>
        public string Model { get; set; } = string.Empty;

        /// <summary>
        /// Response choices (typically one for non-streaming)
        /// </summary>
        public List<ResponseChoice> Choices { get; set; } = new();

        /// <summary>
        /// Token usage information
        /// </summary>
        public UsageInfo Usage { get; set; } = new();

        /// <summary>
        /// Provider-specific data (escape hatch)
        /// </summary>
        public Dictionary<string, object>? ProviderSpecific { get; set; }
    }

    /// <summary>
    /// A single response choice
    /// </summary>
    public class ResponseChoice
    {
        /// <summary>
        /// Index of this choice
        /// </summary>
        public int Index { get; set; }

        /// <summary>
        /// The generated message
        /// </summary>
        public UnifiedMessage Message { get; set; } = new();

        /// <summary>
        /// Why the generation stopped
        /// </summary>
        public FinishReason FinishReason { get; set; }
    }

    /// <summary>
    /// Standardized finish reasons
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum FinishReason
    {
        Stop,          // Natural stop
        MaxTokens,     // Hit token limit
        ToolCalls,     // Stopped to execute tools
        ContentFilter, // Content filtered
        Error,         // Error occurred
        Other          // Other/unknown reason
    }

    /// <summary>
    /// Token usage information
    /// </summary>
    public class UsageInfo
    {
        public int PromptTokens { get; set; }
        public int CompletionTokens { get; set; }
        public int TotalTokens { get; set; }

        // Optional cache-related tokens (for Claude)
        public int? CacheCreationTokens { get; set; }
        public int? CacheReadTokens { get; set; }
    }
}
