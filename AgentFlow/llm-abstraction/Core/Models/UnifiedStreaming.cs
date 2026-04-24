using System.Collections.Generic;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// Represents a chunk of streamed response
    /// </summary>
    public class StreamChunk
    {
        /// <summary>
        /// Unique identifier for this chunk (may be same across chunks in one response)
        /// </summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>
        /// Model that generated this chunk
        /// </summary>
        public string Model { get; set; } = string.Empty;

        /// <summary>
        /// Index of the choice (for n > 1)
        /// </summary>
        public int ChoiceIndex { get; set; }

        /// <summary>
        /// Delta content for this chunk
        /// </summary>
        public StreamDelta Delta { get; set; } = new();

        /// <summary>
        /// Finish reason (only present in final chunk)
        /// </summary>
        public FinishReason? FinishReason { get; set; }

        /// <summary>
        /// Usage information (only present in final chunk for some providers)
        /// </summary>
        public UsageInfo? Usage { get; set; }
    }

    /// <summary>
    /// Delta content in a stream chunk
    /// </summary>
    public class StreamDelta
    {
        /// <summary>
        /// Role (usually only in first chunk)
        /// </summary>
        public MessageRole? Role { get; set; }

        /// <summary>
        /// Text content delta
        /// </summary>
        public string? Content { get; set; }

        /// <summary>
        /// Tool call deltas
        /// </summary>
        public List<ToolCallDelta>? ToolCalls { get; set; }
    }

    /// <summary>
    /// Represents a delta for a tool call in streaming
    /// </summary>
    public class ToolCallDelta
    {
        /// <summary>
        /// Index of the tool call
        /// </summary>
        public int Index { get; set; }

        /// <summary>
        /// Tool call ID (may be in first chunk only)
        /// </summary>
        public string? Id { get; set; }

        /// <summary>
        /// Tool/function name (may be in first chunk only)
        /// </summary>
        public string? Name { get; set; }

        /// <summary>
        /// Arguments delta (JSON string fragment)
        /// </summary>
        public string? Arguments { get; set; }
    }
}
