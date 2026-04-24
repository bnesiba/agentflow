using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.Claude.Models
{
    /// <summary>
    /// Claude Message Request
    /// </summary>
    public class ClaudeMessageRequest
    {
        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("max_tokens")]
        public int MaxTokens { get; set; }  // Required by Claude

        [JsonPropertyName("messages")]
        public List<ClaudeMessage> Messages { get; set; } = new();

        [JsonPropertyName("system")]
        public string? System { get; set; }

        [JsonPropertyName("temperature")]
        public double? Temperature { get; set; }

        [JsonPropertyName("top_p")]
        public double? TopP { get; set; }

        [JsonPropertyName("top_k")]
        public int? TopK { get; set; }

        [JsonPropertyName("stop_sequences")]
        public List<string>? StopSequences { get; set; }

        [JsonPropertyName("stream")]
        public bool Stream { get; set; }

        [JsonPropertyName("tools")]
        public List<ClaudeTool>? Tools { get; set; }

        [JsonPropertyName("tool_choice")]
        public object? ToolChoice { get; set; }

        [JsonPropertyName("output_config")]
        public ClaudeOutputConfig? OutputConfig { get; set; }
    }

    /// <summary>
    /// Claude Message
    /// </summary>
    public class ClaudeMessage
    {
        [JsonPropertyName("role")]
        public string Role { get; set; } = string.Empty;

        [JsonPropertyName("content")]
        public object Content { get; set; } = string.Empty;  // Can be string or array
    }

    /// <summary>
    /// Claude Tool Definition
    /// </summary>
    public class ClaudeTool
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string Description { get; set; } = string.Empty;

        [JsonPropertyName("input_schema")]
        public Dictionary<string, object> InputSchema { get; set; } = new();
    }

    /// <summary>
    /// Claude Message Response
    /// </summary>
    public class ClaudeMessageResponse
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("role")]
        public string Role { get; set; } = string.Empty;

        [JsonPropertyName("content")]
        public List<ClaudeContentBlock> Content { get; set; } = new();

        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("stop_reason")]
        public string? StopReason { get; set; }

        [JsonPropertyName("usage")]
        public ClaudeUsage Usage { get; set; } = new();
    }

    /// <summary>
    /// Claude Content Block
    /// </summary>
    public class ClaudeContentBlock
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("id")]
        public string? Id { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("input")]
        public Dictionary<string, object>? Input { get; set; }
    }

    /// <summary>
    /// Claude Usage Information
    /// </summary>
    public class ClaudeUsage
    {
        [JsonPropertyName("input_tokens")]
        public int InputTokens { get; set; }

        [JsonPropertyName("output_tokens")]
        public int OutputTokens { get; set; }

        [JsonPropertyName("cache_creation_input_tokens")]
        public int? CacheCreationInputTokens { get; set; }

        [JsonPropertyName("cache_read_input_tokens")]
        public int? CacheReadInputTokens { get; set; }
    }

    /// <summary>
    /// Claude Output Config
    /// </summary>
    public class ClaudeOutputConfig
    {
        [JsonPropertyName("format")]
        public ClaudeOutputFormat? Format { get; set; }
    }

    /// <summary>
    /// Claude Output Format
    /// </summary>
    public class ClaudeOutputFormat
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("schema")]
        public Dictionary<string, object>? Schema { get; set; }
    }
}
