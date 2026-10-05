using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Core.Interfaces
{
    /// <summary>
    /// Lifecycle operations for provider-managed explicit prompt-cache resources.
    /// Providers whose caching is request-scoped or implicit do not implement this interface.
    /// </summary>
    public interface IPromptCacheService
    {
        Task<PromptCacheResource> CreatePromptCacheAsync(
            PromptCacheCreateRequest request,
            CancellationToken cancellationToken = default);

        Task<PromptCacheResource> GetPromptCacheAsync(
            string name,
            CancellationToken cancellationToken = default);

        Task<PromptCacheResourcePage> ListPromptCachesAsync(
            int? pageSize = null,
            string? pageToken = null,
            CancellationToken cancellationToken = default);

        Task<PromptCacheResource> UpdatePromptCacheExpirationAsync(
            string name,
            PromptCacheExpiration expiration,
            CancellationToken cancellationToken = default);

        Task DeletePromptCacheAsync(
            string name,
            CancellationToken cancellationToken = default);
    }
}
