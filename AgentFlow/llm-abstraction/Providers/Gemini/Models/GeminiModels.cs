using System.Collections.Generic;
using System.Text.Json;
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

        [JsonPropertyName("safetySettings")]
        public List<Dictionary<string, object>>? SafetySettings { get; set; }

        [JsonPropertyName("cachedContent")]
        public string? CachedContent { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
    }

    public sealed class GeminiCountTokensRequest
    {
        [JsonPropertyName("generateContentRequest")]
        public GeminiGenerateRequest GenerateContentRequest { get; set; } = new();
    }

    public sealed class GeminiCachedContent
    {
        [JsonPropertyName("contents")]
        public List<GeminiContent>? Contents { get; set; }

        [JsonPropertyName("tools")]
        public List<GeminiTool>? Tools { get; set; }

        [JsonPropertyName("systemInstruction")]
        public GeminiContent? SystemInstruction { get; set; }

        [JsonPropertyName("toolConfig")]
        public GeminiToolConfig? ToolConfig { get; set; }

        [JsonPropertyName("ttl")]
        public string? Ttl { get; set; }

        [JsonPropertyName("expireTime")]
        public DateTimeOffset? ExpireTime { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("displayName")]
        public string? DisplayName { get; set; }

        [JsonPropertyName("model")]
        public string? Model { get; set; }

        [JsonPropertyName("createTime")]
        public DateTimeOffset? CreateTime { get; set; }

        [JsonPropertyName("updateTime")]
        public DateTimeOffset? UpdateTime { get; set; }

        [JsonPropertyName("usageMetadata")]
        public GeminiCachedContentUsage? UsageMetadata { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
    }

    public sealed class GeminiCachedContentUsage
    {
        [JsonPropertyName("totalTokenCount")]
        public int? TotalTokenCount { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
    }

    public sealed class GeminiCachedContentList
    {
        [JsonPropertyName("cachedContents")]
        public List<GeminiCachedContent> CachedContents { get; set; } = new();

        [JsonPropertyName("nextPageToken")]
        public string? NextPageToken { get; set; }
    }

    public sealed class GeminiCountTokensResponse
    {
        [JsonPropertyName("totalTokens")]
        public int TotalTokens { get; set; }

        [JsonPropertyName("cachedContentTokenCount")]
        public int? CachedContentTokenCount { get; set; }

        [JsonPropertyName("promptTokensDetails")]
        public List<GeminiModalityTokenCount>? PromptTokensDetails { get; set; }

        [JsonPropertyName("cacheTokensDetails")]
        public List<GeminiModalityTokenCount>? CacheTokensDetails { get; set; }
    }

    public sealed class GeminiModalityTokenCount
    {
        [JsonPropertyName("modality")]
        public string? Modality { get; set; }

        [JsonPropertyName("tokenCount")]
        public int TokenCount { get; set; }
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

        [JsonPropertyName("fileData")]
        public GeminiFileData? FileData { get; set; }

        [JsonPropertyName("thoughtSignature")]
        public string? ThoughtSignature { get; set; }

        [JsonExtensionData]
        public Dictionary<string, JsonElement>? AdditionalProperties { get; set; }
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
    /// Gemini file data part.
    /// </summary>
    public class GeminiFileData
    {
        [JsonPropertyName("mimeType")]
        public string MimeType { get; set; } = string.Empty;

        [JsonPropertyName("fileUri")]
        public string FileUri { get; set; } = string.Empty;

        [JsonPropertyName("displayName")]
        public string? DisplayName { get; set; }
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

        [JsonPropertyName("id")]
        public string? Id { get; set; }
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

        [JsonPropertyName("id")]
        public string? Id { get; set; }
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

        [JsonPropertyName("responseJsonSchema")]
        public Dictionary<string, object>? ResponseJsonSchema { get; set; }

        [JsonPropertyName("thinkingConfig")]
        public GeminiThinkingConfig? ThinkingConfig { get; set; }
    }

    /// <summary>
    /// Gemini thinking configuration.
    /// </summary>
    public class GeminiThinkingConfig
    {
        [JsonPropertyName("includeThoughts")]
        public bool? IncludeThoughts { get; set; }

        [JsonPropertyName("thinkingBudget")]
        public int? ThinkingBudget { get; set; }

        [JsonPropertyName("thinkingLevel")]
        public string? ThinkingLevel { get; set; }
    }

    /// <summary>
    /// Gemini Tool
    /// </summary>
    public class GeminiTool
    {
        [JsonPropertyName("functionDeclarations")]
        public List<GeminiFunctionDeclaration>? FunctionDeclarations { get; set; }

        [JsonPropertyName("googleSearch")]
        public GeminiGoogleSearch? GoogleSearch { get; set; }
    }

    public sealed class GeminiGoogleSearch
    {
        [JsonPropertyName("timeRangeFilter")]
        public GeminiTimeRangeFilter? TimeRangeFilter { get; set; }

        [JsonPropertyName("searchTypes")]
        public GeminiSearchTypes? SearchTypes { get; set; }
    }

    public sealed class GeminiTimeRangeFilter
    {
        [JsonPropertyName("startTime")]
        public string StartTime { get; set; } = string.Empty;

        [JsonPropertyName("endTime")]
        public string EndTime { get; set; } = string.Empty;
    }

    public sealed class GeminiSearchTypes
    {
        [JsonPropertyName("webSearch")]
        public Dictionary<string, object>? WebSearch { get; set; }

        [JsonPropertyName("imageSearch")]
        public Dictionary<string, object>? ImageSearch { get; set; }
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

        [JsonPropertyName("allowedFunctionNames")]
        public List<string>? AllowedFunctionNames { get; set; }
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

        [JsonPropertyName("modelVersion")]
        public string? ModelVersion { get; set; }

        [JsonPropertyName("responseId")]
        public string? ResponseId { get; set; }

        [JsonPropertyName("promptFeedback")]
        public Dictionary<string, object>? PromptFeedback { get; set; }
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

        [JsonPropertyName("groundingMetadata")]
        public JsonElement? GroundingMetadata { get; set; }
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

        [JsonPropertyName("thoughtsTokenCount")]
        public int? ThoughtsTokenCount { get; set; }

        [JsonPropertyName("cachedContentTokenCount")]
        public int? CachedContentTokenCount { get; set; }
    }
}
