using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.OpenAI.Models
{
    /// <summary>
    /// OpenAI Streaming Response Chunk
    /// </summary>
    public class OpenAIStreamChunk
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        [JsonPropertyName("object")]
        public string Object { get; set; } = string.Empty;

        [JsonPropertyName("created")]
        public long Created { get; set; }

        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("choices")]
        public List<OpenAIStreamChoice> Choices { get; set; } = new();

        [JsonPropertyName("usage")]
        public OpenAIUsage? Usage { get; set; }
    }

    /// <summary>
    /// OpenAI Stream Choice
    /// </summary>
    public class OpenAIStreamChoice
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("delta")]
        public OpenAIDelta Delta { get; set; } = new();

        [JsonPropertyName("finish_reason")]
        public string? FinishReason { get; set; }
    }

    /// <summary>
    /// OpenAI Delta
    /// </summary>
    public class OpenAIDelta
    {
        [JsonPropertyName("role")]
        public string? Role { get; set; }

        [JsonPropertyName("content")]
        public string? Content { get; set; }

        [JsonPropertyName("tool_calls")]
        public List<OpenAIToolCallDelta>? ToolCalls { get; set; }
    }

    /// <summary>
    /// OpenAI Tool Call Delta
    /// </summary>
    public class OpenAIToolCallDelta
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("id")]
        public string? Id { get; set; }

        [JsonPropertyName("type")]
        public string? Type { get; set; }

        [JsonPropertyName("function")]
        public OpenAIFunctionDelta? Function { get; set; }
    }

    /// <summary>
    /// OpenAI Function Delta
    /// </summary>
    public class OpenAIFunctionDelta
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("arguments")]
        public string? Arguments { get; set; }
    }
}
