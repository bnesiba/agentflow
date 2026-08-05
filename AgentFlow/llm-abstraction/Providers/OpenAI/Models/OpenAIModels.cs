using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.OpenAI.Models
{
    /// <summary>
    /// OpenAI Responses API request.
    /// </summary>
    public class OpenAIResponseRequest
    {
        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("input")]
        public List<object> Input { get; set; } = new();

        [JsonPropertyName("previous_response_id")]
        public string? PreviousResponseId { get; set; }

        [JsonPropertyName("instructions")]
        public string? Instructions { get; set; }

        [JsonPropertyName("max_output_tokens")]
        public int? MaxOutputTokens { get; set; }

        [JsonPropertyName("temperature")]
        public double? Temperature { get; set; }

        [JsonPropertyName("top_p")]
        public double? TopP { get; set; }

        [JsonPropertyName("stream")]
        public bool Stream { get; set; }

        [JsonPropertyName("tools")]
        public List<OpenAIResponseTool>? Tools { get; set; }

        [JsonPropertyName("tool_choice")]
        public object? ToolChoice { get; set; }

        [JsonPropertyName("parallel_tool_calls")]
        public bool? ParallelToolCalls { get; set; }

        [JsonPropertyName("text")]
        public OpenAITextConfig? Text { get; set; }

        [JsonPropertyName("reasoning")]
        public OpenAIReasoningConfig? Reasoning { get; set; }

        [JsonPropertyName("user")]
        public string? User { get; set; }

        [JsonPropertyName("metadata")]
        public Dictionary<string, string>? Metadata { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
    }

    public class OpenAIResponseTool
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = "function";

        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("parameters")]
        public Dictionary<string, object> Parameters { get; set; } = new();

        [JsonPropertyName("strict")]
        public bool Strict { get; set; } = true;
    }

    public class OpenAITextConfig
    {
        [JsonPropertyName("format")]
        public object? Format { get; set; }
    }

    public class OpenAIReasoningConfig
    {
        [JsonPropertyName("effort")]
        public string? Effort { get; set; }

        [JsonPropertyName("summary")]
        public string? Summary { get; set; }
    }

    /// <summary>
    /// OpenAI Responses API response.
    /// </summary>
    public class OpenAIResponse
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("status")]
        public string? Status { get; set; }

        [JsonPropertyName("output")]
        public List<OpenAIOutputItem> Output { get; set; } = new();

        [JsonPropertyName("usage")]
        public OpenAIResponseUsage? Usage { get; set; }

        [JsonPropertyName("incomplete_details")]
        public OpenAIIncompleteDetails? IncompleteDetails { get; set; }

        [JsonPropertyName("error")]
        public OpenAIResponseError? Error { get; set; }
    }

    public class OpenAIOutputItem
    {
        [JsonPropertyName("id")]
        public string? Id { get; set; }

        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("status")]
        public string? Status { get; set; }

        [JsonPropertyName("role")]
        public string? Role { get; set; }

        [JsonPropertyName("content")]
        public List<OpenAIOutputContent>? Content { get; set; }

        [JsonPropertyName("call_id")]
        public string? CallId { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("arguments")]
        public string? Arguments { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? ExtensionData { get; set; }
    }

    public class OpenAIOutputContent
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("refusal")]
        public string? Refusal { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? ExtensionData { get; set; }
    }

    public class OpenAIResponseUsage
    {
        [JsonPropertyName("input_tokens")]
        public int InputTokens { get; set; }

        [JsonPropertyName("output_tokens")]
        public int OutputTokens { get; set; }

        [JsonPropertyName("total_tokens")]
        public int TotalTokens { get; set; }

        [JsonPropertyName("input_tokens_details")]
        public OpenAIInputTokenDetails? InputTokenDetails { get; set; }

        [JsonPropertyName("output_tokens_details")]
        public OpenAIOutputTokenDetails? OutputTokenDetails { get; set; }
    }

    public class OpenAIInputTokenDetails
    {
        [JsonPropertyName("cached_tokens")]
        public int? CachedTokens { get; set; }
    }

    public class OpenAIOutputTokenDetails
    {
        [JsonPropertyName("reasoning_tokens")]
        public int? ReasoningTokens { get; set; }
    }

    public class OpenAIIncompleteDetails
    {
        [JsonPropertyName("reason")]
        public string? Reason { get; set; }
    }

    public class OpenAIResponseError
    {
        [JsonPropertyName("code")]
        public string? Code { get; set; }

        [JsonPropertyName("message")]
        public string? Message { get; set; }
    }
}
