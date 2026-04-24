using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Providers.Gemini.Models
{
    /// <summary>
    /// Gemini Streaming Response (same structure as regular response, but sent incrementally)
    /// </summary>
    public class GeminiStreamChunk
    {
        [JsonPropertyName("candidates")]
        public List<GeminiCandidate>? Candidates { get; set; }

        [JsonPropertyName("usageMetadata")]
        public GeminiUsageMetadata? UsageMetadata { get; set; }

        [JsonPropertyName("modelVersion")]
        public string? ModelVersion { get; set; }
    }
}
