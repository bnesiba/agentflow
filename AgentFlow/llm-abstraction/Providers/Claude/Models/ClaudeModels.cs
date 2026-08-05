using System.Text.Json;
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
        public object? System { get; set; }

        [JsonPropertyName("cache_control")]
        public ClaudeCacheControl? CacheControl { get; set; }

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

        [JsonPropertyName("thinking")]
        public ClaudeThinkingConfig? Thinking { get; set; }

        [JsonPropertyName("metadata")]
        public ClaudeMetadata? Metadata { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
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
        [JsonPropertyName("type")]
        public string? Type { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("tool_use_id")]
        public string? ToolUseId { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("input_schema")]
        public Dictionary<string, object>? InputSchema { get; set; }

        [JsonPropertyName("strict")]
        public bool? Strict { get; set; }

        [JsonPropertyName("max_uses")]
        public int? MaxUses { get; set; }

        [JsonPropertyName("allowed_domains")]
        public IReadOnlyList<string>? AllowedDomains { get; set; }

        [JsonPropertyName("blocked_domains")]
        public IReadOnlyList<string>? BlockedDomains { get; set; }

        [JsonPropertyName("user_location")]
        public Dictionary<string, object>? UserLocation { get; set; }

        [JsonPropertyName("allowed_callers")]
        public List<string>? AllowedCallers { get; set; }

        [JsonPropertyName("response_inclusion")]
        public string? ResponseInclusion { get; set; }

        [JsonPropertyName("cache_control")]
        public ClaudeCacheControl? CacheControl { get; set; }
    }

    public sealed class ClaudeCacheControl
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = "ephemeral";

        [JsonPropertyName("ttl")]
        public string? Ttl { get; set; }
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

        [JsonPropertyName("stop_sequence")]
        public string? StopSequence { get; set; }

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

        [JsonPropertyName("tool_use_id")]
        public string? ToolUseId { get; set; }

        [JsonPropertyName("input")]
        public Dictionary<string, object>? Input { get; set; }

        [JsonPropertyName("thinking")]
        public string? Thinking { get; set; }

        [JsonPropertyName("signature")]
        public string? Signature { get; set; }

        [JsonPropertyName("data")]
        public string? Data { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
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

        [JsonPropertyName("ephemeral_5m_input_tokens")]
        public int? Ephemeral5mInputTokens { get; set; }

        [JsonPropertyName("ephemeral_1h_input_tokens")]
        public int? Ephemeral1hInputTokens { get; set; }

        [JsonPropertyName("cache_creation")]
        public ClaudeCacheCreationUsage? CacheCreation { get; set; }

        [JsonPropertyName("server_tool_use")]
        public Dictionary<string, int>? ServerToolUse { get; set; }
    }

    public sealed class ClaudeCacheCreationUsage
    {
        [JsonPropertyName("ephemeral_5m_input_tokens")]
        public int? Ephemeral5mInputTokens { get; set; }

        [JsonPropertyName("ephemeral_1h_input_tokens")]
        public int? Ephemeral1hInputTokens { get; set; }
    }

    public sealed class ClaudeMessageTokensCount
    {
        [JsonPropertyName("input_tokens")]
        public int InputTokens { get; set; }
    }

    /// <summary>
    /// Claude Output Config
    /// </summary>
    public class ClaudeOutputConfig
    {
        [JsonPropertyName("format")]
        public ClaudeOutputFormat? Format { get; set; }

        [JsonPropertyName("effort")]
        public string? Effort { get; set; }
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

    /// <summary>
    /// Claude thinking configuration.
    /// </summary>
    public class ClaudeThinkingConfig
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("budget_tokens")]
        public int? BudgetTokens { get; set; }

        [JsonPropertyName("display")]
        public string? Display { get; set; }
    }

    /// <summary>
    /// Claude request metadata.
    /// </summary>
    public class ClaudeMetadata
    {
        [JsonPropertyName("user_id")]
        public string? UserId { get; set; }
    }
}
