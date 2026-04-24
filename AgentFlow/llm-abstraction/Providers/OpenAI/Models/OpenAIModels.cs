using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.OpenAI.Models
{
    /// <summary>
    /// OpenAI Chat Completion Request
    /// </summary>
    public class OpenAIChatRequest
    {
        [JsonPropertyName("model")]
        public string Model { get; set; } = string.Empty;

        [JsonPropertyName("messages")]
        public List<OpenAIMessage> Messages { get; set; } = new();

        [JsonPropertyName("max_tokens")]
        public int? MaxTokens { get; set; }

        [JsonPropertyName("temperature")]
        public double? Temperature { get; set; }

        [JsonPropertyName("top_p")]
        public double? TopP { get; set; }

        [JsonPropertyName("stop")]
        public List<string>? Stop { get; set; }

        [JsonPropertyName("stream")]
        public bool Stream { get; set; }

        [JsonPropertyName("tools")]
        public List<OpenAITool>? Tools { get; set; }

        [JsonPropertyName("tool_choice")]
        public object? ToolChoice { get; set; }

        [JsonPropertyName("response_format")]
        public object? ResponseFormat { get; set; }
    }

    /// <summary>
    /// OpenAI Message
    /// </summary>
    public class OpenAIMessage
    {
        [JsonPropertyName("role")]
        public string Role { get; set; } = string.Empty;

        [JsonPropertyName("content")]
        public object? Content { get; set; }  // Can be string or array of content parts

        [JsonPropertyName("tool_calls")]
        public List<OpenAIToolCall>? ToolCalls { get; set; }

        [JsonPropertyName("tool_call_id")]
        public string? ToolCallId { get; set; }
    }

    /// <summary>
    /// OpenAI Tool Definition
    /// </summary>
    public class OpenAITool
    {
        [JsonPropertyName("type")]
        public string Type { get; set; } = "function";

        [JsonPropertyName("function")]
        public OpenAIFunction Function { get; set; } = new();
    }

    /// <summary>
    /// OpenAI Function Definition
    /// </summary>
    public class OpenAIFunction
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string Description { get; set; } = string.Empty;

        [JsonPropertyName("parameters")]
        public Dictionary<string, object> Parameters { get; set; } = new();
    }

    /// <summary>
    /// OpenAI Tool Call (in assistant message)
    /// </summary>
    public class OpenAIToolCall
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = string.Empty;

        [JsonPropertyName("type")]
        public string Type { get; set; } = "function";

        [JsonPropertyName("function")]
        public OpenAIFunctionCall Function { get; set; } = new();
    }

    /// <summary>
    /// OpenAI Function Call
    /// </summary>
    public class OpenAIFunctionCall
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("arguments")]
        public string Arguments { get; set; } = string.Empty;  // JSON string
    }

    /// <summary>
    /// OpenAI Chat Completion Response
    /// </summary>
    public class OpenAIChatResponse
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
        public List<OpenAIChoice> Choices { get; set; } = new();

        [JsonPropertyName("usage")]
        public OpenAIUsage Usage { get; set; } = new();
    }

    /// <summary>
    /// OpenAI Choice
    /// </summary>
    public class OpenAIChoice
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("message")]
        public OpenAIMessage Message { get; set; } = new();

        [JsonPropertyName("finish_reason")]
        public string? FinishReason { get; set; }
    }

    /// <summary>
    /// OpenAI Usage Information
    /// </summary>
    public class OpenAIUsage
    {
        [JsonPropertyName("prompt_tokens")]
        public int PromptTokens { get; set; }

        [JsonPropertyName("completion_tokens")]
        public int CompletionTokens { get; set; }

        [JsonPropertyName("total_tokens")]
        public int TotalTokens { get; set; }
    }
}
