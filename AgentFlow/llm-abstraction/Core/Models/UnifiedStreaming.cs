using System.Collections.Generic;
using LLMAbstraction.Core.Transport;

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

        public Dictionary<string, object>? ProviderMetadata { get; set; }

        /// <summary>
        /// Provider-bound continuation state when the streaming provider supplies
        /// a complete final response object.
        /// </summary>
        public ProviderContinuationState? Continuation { get; set; }

        /// <summary>
        /// Fully accumulated assistant message when the provider stream reaches a
        /// terminal event. This can be appended to conversation history without
        /// reconstructing native signed blocks from deltas.
        /// </summary>
        public UnifiedMessage? CompletedMessage { get; set; }

        /// <summary>
        /// HTTP attempt, request ID, and rate-limit information. Providers attach
        /// this to the first visible stream chunk.
        /// </summary>
        public TransportMetadata? Transport { get; set; }
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

        /// <summary>
        /// Complete provider-native content blocks observed in this event. Text in
        /// <see cref="Content"/> remains the display-oriented incremental delta.
        /// </summary>
        public List<ContentBlock>? ContentBlocks { get; set; }
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

        /// <summary>
        /// Complete arguments JSON when the tool call has finished streaming.
        /// </summary>
        public string? CompleteArguments { get; set; }

        /// <summary>
        /// True when the provider has completed this tool call.
        /// </summary>
        public bool IsComplete { get; set; }

        /// <summary>
        /// Provider response item identifier when it differs from the call ID.
        /// </summary>
        public string? ItemId { get; set; }

        public ProviderNativeRepresentation? NativeRepresentation { get; set; }

        public string? Type { get; set; }
    }
}
