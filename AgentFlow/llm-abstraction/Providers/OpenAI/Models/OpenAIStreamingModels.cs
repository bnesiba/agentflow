using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.OpenAI.Models
{
    /// <summary>
    /// Generic OpenAI Responses API streaming event.
    /// </summary>
    public class OpenAIResponseStreamEvent
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = string.Empty;

        [JsonPropertyName("response")]
        public OpenAIResponse? Response { get; set; }

        [JsonPropertyName("response_id")]
        public string? ResponseId { get; set; }

        [JsonPropertyName("item_id")]
        public string? ItemId { get; set; }

        [JsonPropertyName("output_index")]
        public int? OutputIndex { get; set; }

        [JsonPropertyName("delta")]
        public string? Delta { get; set; }

        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("arguments")]
        public string? Arguments { get; set; }

        [JsonPropertyName("sequence_number")]
        public int? SequenceNumber { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? ExtensionData { get; set; }
    }
}
