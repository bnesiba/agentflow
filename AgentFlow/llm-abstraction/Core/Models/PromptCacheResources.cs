using System.Text.Json;
using LLMAbstraction.Core.Transport;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// A stable prompt prefix used to create a provider-managed cache resource.
    /// Generation-only fields on Prefix are ignored by resource creation.
    /// </summary>
    public sealed class PromptCacheCreateRequest
    {
        public required UnifiedRequest Prefix { get; init; }
        public string? DisplayName { get; init; }
        public TimeSpan? Ttl { get; init; }
        public DateTimeOffset? ExpireTime { get; init; }
        public RequestTransportOptions? Transport { get; init; }
    }

    public sealed class PromptCacheExpiration
    {
        public TimeSpan? Ttl { get; init; }
        public DateTimeOffset? ExpireTime { get; init; }
    }

    public sealed class PromptCacheResource
    {
        public required LLMProvider Provider { get; init; }
        public required string Name { get; init; }
        public required string Model { get; init; }
        public string? DisplayName { get; init; }
        public DateTimeOffset? CreateTime { get; init; }
        public DateTimeOffset? UpdateTime { get; init; }
        public DateTimeOffset? ExpireTime { get; init; }
        public int? CachedTokens { get; init; }
        public JsonElement Native { get; init; }
        public TransportMetadata? Transport { get; init; }
    }

    public sealed class PromptCacheResourcePage
    {
        public IReadOnlyList<PromptCacheResource> Items { get; init; } = Array.Empty<PromptCacheResource>();
        public string? NextPageToken { get; init; }
        public TransportMetadata? Transport { get; init; }
    }
}
