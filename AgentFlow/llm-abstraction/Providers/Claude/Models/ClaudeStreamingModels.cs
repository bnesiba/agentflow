using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.Claude.Models
{
    /// <summary>
    /// Claude Streaming Event
    /// </summary>
    public class ClaudeStreamEvent
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;
    }

    /// <summary>
    /// Claude Message Start Event
    /// </summary>
    public class ClaudeMessageStart : ClaudeStreamEvent
    {
        [JsonPropertyName("message")]
        public ClaudeMessageResponse? Message { get; set; }
    }

    /// <summary>
    /// Claude Content Block Start Event
    /// </summary>
    public class ClaudeContentBlockStart : ClaudeStreamEvent
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("content_block")]
        public ClaudeContentBlock? ContentBlock { get; set; }
    }

    /// <summary>
    /// Claude Content Block Delta Event
    /// </summary>
    public class ClaudeContentBlockDelta : ClaudeStreamEvent
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("delta")]
        public ClaudeDelta? Delta { get; set; }
    }

    /// <summary>
    /// Claude Delta
    /// </summary>
    public class ClaudeDelta
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("partial_json")]
        public string? PartialJson { get; set; }
    }

    /// <summary>
    /// Claude Message Delta Event
    /// </summary>
    public class ClaudeMessageDelta : ClaudeStreamEvent
    {
        [JsonPropertyName("delta")]
        public ClaudeMessageDeltaData? Delta { get; set; }

        [JsonPropertyName("usage")]
        public ClaudeUsage? Usage { get; set; }
    }

    /// <summary>
    /// Claude Message Delta Data
    /// </summary>
    public class ClaudeMessageDeltaData
    {
        [JsonPropertyName("stop_reason")]
        public string? StopReason { get; set; }

        [JsonPropertyName("stop_sequence")]
        public string? StopSequence { get; set; }
    }
}
