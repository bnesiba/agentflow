using System.Text.Json;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.Gemini.Models
{
    public sealed class GeminiInteractionRequest
    {
        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("input")]
        public List<JsonElement> Input { get; set; } = new();

        [JsonPropertyName("system_instruction")]
        public string? SystemInstruction { get; set; }

        [JsonPropertyName("tools")]
        public List<JsonElement>? Tools { get; set; }

        [JsonPropertyName("response_format")]
        public List<JsonElement>? ResponseFormat { get; set; }

        [JsonPropertyName("stream")]
        public bool? Stream { get; set; }

        [JsonPropertyName("store")]
        public bool Store { get; set; }

        [JsonPropertyName("previous_interaction_id")]
        public string? PreviousInteractionId { get; set; }

        [JsonPropertyName("generation_config")]
        public Dictionary<string, object>? GenerationConfig { get; set; }

        [JsonPropertyName("safety_settings")]
        public object? SafetySettings { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
    }

    public sealed class GeminiInteractionResponse
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("status")]
        public string? Status { get; set; }

        [JsonPropertyName("steps")]
        public List<JsonElement> Steps { get; set; } = new();

        [JsonPropertyName("usage")]
        public GeminiInteractionUsage? Usage { get; set; }

        [JsonPropertyName("error")]
        public JsonElement? Error { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
    }

    public sealed class GeminiInteractionUsage
    {
        [JsonPropertyName("total_input_tokens")]
        public int TotalInputTokens { get; set; }

        [JsonPropertyName("total_output_tokens")]
        public int TotalOutputTokens { get; set; }

        [JsonPropertyName("total_tokens")]
        public int TotalTokens { get; set; }

        [JsonPropertyName("total_thought_tokens")]
        public int? TotalThoughtTokens { get; set; }

        [JsonPropertyName("total_tool_use_tokens")]
        public int? TotalToolUseTokens { get; set; }

        [JsonPropertyName("total_cached_tokens")]
        public int? TotalCachedTokens { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
    }

    public sealed class GeminiInteractionStreamEvent
    {
        [JsonPropertyName("event_type")]
        public string EventType { get; set; } = string.Empty;

        [JsonPropertyName("index")]
        public int? Index { get; set; }

        [JsonPropertyName("step")]
        public JsonElement? Step { get; set; }

        [JsonPropertyName("delta")]
        public JsonElement? Delta { get; set; }

        [JsonPropertyName("interaction")]
        public GeminiInteractionResponse? Interaction { get; set; }

        [JsonPropertyName("error")]
        public JsonElement? Error { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
    }
}
