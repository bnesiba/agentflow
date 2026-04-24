using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.Gemini.Models
{
    /// <summary>
    /// Gemini Generate Content Request
    /// </summary>
    public class GeminiGenerateRequest
    {
        [JsonPropertyName("contents")]
        public List<GeminiContent> Contents { get; set; } = new();

        [JsonPropertyName("systemInstruction")]
        public GeminiContent? SystemInstruction { get; set; }

        [JsonPropertyName("generationConfig")]
        public GeminiGenerationConfig? GenerationConfig { get; set; }

        [JsonPropertyName("tools")]
        public List<GeminiTool>? Tools { get; set; }

        [JsonPropertyName("toolConfig")]
        public GeminiToolConfig? ToolConfig { get; set; }
    }

    /// <summary>
    /// Gemini Content (message)
    /// </summary>
    public class GeminiContent
    {
        [JsonPropertyName("role")]
        public string? Role { get; set; }

        [JsonPropertyName("parts")]
        public List<GeminiPart> Parts { get; set; } = new();
    }

    /// <summary>
    /// Gemini Part (content piece)
    /// </summary>
    public class GeminiPart
    {
        [JsonPropertyName("text")]
        public string? Text { get; set; }

        [JsonPropertyName("inlineData")]
        public GeminiInlineData? InlineData { get; set; }

        [JsonPropertyName("functionCall")]
        public GeminiFunctionCall? FunctionCall { get; set; }

        [JsonPropertyName("functionResponse")]
        public GeminiFunctionResponse? FunctionResponse { get; set; }
    }

    /// <summary>
    /// Gemini Inline Data (for images, etc.)
    /// </summary>
    public class GeminiInlineData
    {
        [JsonPropertyName("mimeType")]
        public string MimeType { get; set; } = string.Empty;

        [JsonPropertyName("data")]
        public string Data { get; set; } = string.Empty;  // Base64
    }

    /// <summary>
    /// Gemini Function Call
    /// </summary>
    public class GeminiFunctionCall
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("args")]
        public Dictionary<string, object>? Args { get; set; }
    }

    /// <summary>
    /// Gemini Function Response
    /// </summary>
    public class GeminiFunctionResponse
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("response")]
        public Dictionary<string, object> Response { get; set; } = new();
    }

    /// <summary>
    /// Gemini Generation Config
    /// </summary>
    public class GeminiGenerationConfig
    {
        [JsonPropertyName("maxOutputTokens")]
        public int? MaxOutputTokens { get; set; }

        [JsonPropertyName("temperature")]
        public double? Temperature { get; set; }

        [JsonPropertyName("topP")]
        public double? TopP { get; set; }

        [JsonPropertyName("topK")]
        public int? TopK { get; set; }

        [JsonPropertyName("stopSequences")]
        public List<string>? StopSequences { get; set; }

        [JsonPropertyName("responseMimeType")]
        public string? ResponseMimeType { get; set; }

        [JsonPropertyName("responseSchema")]
        public Dictionary<string, object>? ResponseSchema { get; set; }
    }

    /// <summary>
    /// Gemini Tool
    /// </summary>
    public class GeminiTool
    {
        [JsonPropertyName("functionDeclarations")]
        public List<GeminiFunctionDeclaration> FunctionDeclarations { get; set; } = new();
    }

    /// <summary>
    /// Gemini Function Declaration
    /// </summary>
    public class GeminiFunctionDeclaration
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("description")]
        public string Description { get; set; } = string.Empty;

        [JsonPropertyName("parameters")]
        public Dictionary<string, object> Parameters { get; set; } = new();
    }

    /// <summary>
    /// Gemini Tool Config
    /// </summary>
    public class GeminiToolConfig
    {
        [JsonPropertyName("functionCallingConfig")]
        public GeminiFunctionCallingConfig FunctionCallingConfig { get; set; } = new();
    }

    /// <summary>
    /// Gemini Function Calling Config
    /// </summary>
    public class GeminiFunctionCallingConfig
    {
        [JsonPropertyName("mode")]
        public string Mode { get; set; } = "AUTO";  // AUTO, ANY, NONE
    }

    /// <summary>
    /// Gemini Generate Content Response
    /// </summary>
    public class GeminiGenerateResponse
    {
        [JsonPropertyName("candidates")]
        public List<GeminiCandidate> Candidates { get; set; } = new();

        [JsonPropertyName("usageMetadata")]
        public GeminiUsageMetadata? UsageMetadata { get; set; }
    }

    /// <summary>
    /// Gemini Candidate
    /// </summary>
    public class GeminiCandidate
    {
        [JsonPropertyName("content")]
        public GeminiContent Content { get; set; } = new();

        [JsonPropertyName("finishReason")]
        public string? FinishReason { get; set; }

        [JsonPropertyName("index")]
        public int Index { get; set; }

        [JsonPropertyName("safetyRatings")]
        public List<GeminiSafetyRating>? SafetyRatings { get; set; }
    }

    /// <summary>
    /// Gemini Safety Rating
    /// </summary>
    public class GeminiSafetyRating
    {
        [JsonPropertyName("category")]
        public string Category { get; set; } = string.Empty;

        [JsonPropertyName("probability")]
        public string Probability { get; set; } = string.Empty;
    }

    /// <summary>
    /// Gemini Usage Metadata
    /// </summary>
    public class GeminiUsageMetadata
    {
        [JsonPropertyName("promptTokenCount")]
        public int PromptTokenCount { get; set; }

        [JsonPropertyName("candidatesTokenCount")]
        public int CandidatesTokenCount { get; set; }

        [JsonPropertyName("totalTokenCount")]
        public int TotalTokenCount { get; set; }
    }
}
