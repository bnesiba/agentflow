namespace LLMAbstraction.Core.Models
{
    public enum PromptCacheMode
    {
        ProviderDefault,
        PreferReuse,
        ExplicitBreakpointsOnly,
        Disabled
    }

    public enum PromptCacheTtl
    {
        FiveMinutes,
        ThirtyMinutes,
        OneHour,
        TwentyFourHours
    }

    /// <summary>
    /// Marks the end of a stable prompt prefix. Providers only serialize a
    /// directive where that semantic boundary is supported exactly.
    /// </summary>
    public sealed class PromptCacheDirective
    {
        public PromptCacheTtl? Ttl { get; init; }
    }

    public sealed class PromptCacheOptions
    {
        public PromptCacheMode Mode { get; init; } = PromptCacheMode.ProviderDefault;
        public PromptCacheTtl? Ttl { get; init; }
        public OpenAIPromptCacheOptions? OpenAI { get; init; }
        public GeminiPromptCacheOptions? Gemini { get; init; }
    }

    public sealed class OpenAIPromptCacheOptions
    {
        public string? CacheKey { get; init; }
        public OpenAIPromptCacheRetention? Retention { get; init; }
    }

    public enum OpenAIPromptCacheRetention
    {
        InMemory,
        TwentyFourHours
    }

    public sealed class GeminiPromptCacheOptions
    {
        /// <summary>
        /// Provider resource name, for example cachedContents/abc123. Explicit
        /// cache resources are usable only with generateContent.
        /// </summary>
        public string? CachedContentName { get; init; }
    }
}
