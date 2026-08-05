using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Core.Interfaces
{
    /// <summary>
    /// Counts the input tokens in the provider request that would be produced
    /// from a unified generation request.
    /// </summary>
    public interface ITokenCountingService
    {
        Task<TokenCountResult> CountInputTokensAsync(
            UnifiedRequest request,
            CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// A generation service that also exposes its provider's preflight token
    /// counting endpoint without adding endpoint-specific operations to
    /// <see cref="ILLMService"/>.
    /// </summary>
    public interface ILLMServiceWithTokenCounting : ILLMService, ITokenCountingService
    {
    }
}
