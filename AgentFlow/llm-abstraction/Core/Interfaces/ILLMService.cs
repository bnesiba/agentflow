using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Core.Interfaces
{
    /// <summary>
    /// Core interface for LLM service implementations
    /// </summary>
    public interface ILLMService
    {
        /// <summary>
        /// Send a request to the LLM and get a response
        /// </summary>
        Task<UnifiedResponse> GenerateAsync(
            UnifiedRequest request, 
            CancellationToken cancellationToken = default);

        /// <summary>
        /// Stream a response from the LLM
        /// </summary>
        IAsyncEnumerable<StreamChunk> StreamAsync(
            UnifiedRequest request, 
            CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Interface for converting between unified and provider-specific models
    /// </summary>
    /// <typeparam name="TRequest">Provider-specific request type</typeparam>
    /// <typeparam name="TResponse">Provider-specific response type</typeparam>
    public interface IModelConverter<TRequest, TResponse>
    {
        /// <summary>
        /// Convert unified request to provider-specific request
        /// </summary>
        TRequest ConvertRequest(UnifiedRequest request);

        /// <summary>
        /// Convert provider-specific response to unified response
        /// </summary>
        UnifiedResponse ConvertResponse(TResponse response);
    }
}
