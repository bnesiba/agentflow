using System;
using System.Collections.Generic;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace LLMAbstraction.Core.Transport
{
    public sealed class LLMTransportOptions
    {
        public IRequestAdmissionPolicy? AdmissionPolicy { get; init; }
        public TimeProvider TimeProvider { get; init; } = TimeProvider.System;
    }

    /// <summary>
    /// Optional controls for one logical request. Service-wide transport
    /// settings remain the default when these values are omitted.
    /// </summary>
    public sealed class RequestTransportOptions
    {
        public int? EstimatedInputTokens { get; init; }

        internal void Validate()
        {

            if (EstimatedInputTokens < 0)
                throw new ArgumentOutOfRangeException(nameof(EstimatedInputTokens));
        }
    }

    public sealed class TransportAttempt
    {
        public int AttemptNumber { get; init; }
        public DateTimeOffset StartedAt { get; init; }
        public TimeSpan Duration { get; init; }
        public HttpStatusCode? StatusCode { get; init; }
        public string? ExceptionType { get; init; }
        public string? ExceptionMessage { get; init; }
    }

    public sealed class RequestAdmissionContext
    {
        public required LLMProvider Provider { get; init; }
        public required string Endpoint { get; init; }
        public required string Model { get; init; }
        public bool IsStreaming { get; init; }
        public int? EstimatedInputTokens { get; init; }
    }

    /// <summary>
    /// Optional hook for application-owned concurrency, distributed rate-limit,
    /// or token-budget admission. The returned lease is held for the logical
    /// request.
    /// </summary>
    public interface IRequestAdmissionPolicy
    {
        ValueTask<IDisposable?> AcquireAsync(
            RequestAdmissionContext context,
            CancellationToken cancellationToken = default);
    }

    public sealed class RateLimitWindow
    {
        public long? Limit { get; init; }
        public long? Remaining { get; init; }
        public DateTimeOffset? ResetsAt { get; init; }
        public TimeSpan? ResetsAfter { get; init; }
    }

    public sealed class RateLimitSnapshot
    {
        public RateLimitWindow? Requests { get; init; }
        public RateLimitWindow? Tokens { get; init; }
        public RateLimitWindow? InputTokens { get; init; }
        public RateLimitWindow? OutputTokens { get; init; }
        public IReadOnlyDictionary<string, string[]> RawHeaders { get; init; } =
            new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
    }

    public sealed class TransportMetadata
    {
        public string? RequestId { get; init; }
        public IReadOnlyList<TransportAttempt> Attempts { get; init; } = Array.Empty<TransportAttempt>();
        public RateLimitSnapshot? RateLimits { get; init; }
    }
}
