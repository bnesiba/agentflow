using System.Collections.Generic;
using LLMAbstraction.Core.Transport;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// Describes how the provider documents a preflight token count.
    /// </summary>
    public enum TokenCountAccuracy
    {
        Exact,
        Estimate
    }

    /// <summary>
    /// Provider-calculated input token usage for a converted request.
    /// </summary>
    public sealed class TokenCountResult
    {
        public required LLMProvider Provider { get; init; }
        public required string Model { get; init; }
        public int InputTokens { get; init; }
        public TokenCountAccuracy Accuracy { get; init; }
        public int? CachedTokens { get; init; }
        public IReadOnlyDictionary<string, object> ProviderMetadata { get; init; } =
            new Dictionary<string, object>();
        public TransportMetadata? Transport { get; init; }
    }
}
