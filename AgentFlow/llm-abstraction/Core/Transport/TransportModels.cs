using System;
using System.Collections.Generic;
using System.Net;
using System.Threading;
using System.Threading.Tasks;

namespace LLMAbstraction.Core.Transport
{
    public sealed class LLMTransportOptions
    {
        public RetryPolicy Retry { get; init; } = new();
        public IRetryObserver? RetryObserver { get; init; }
        public IRequestAdmissionPolicy? AdmissionPolicy { get; init; }
        public IRetryDelayScheduler DelayScheduler { get; init; } = SystemRetryDelayScheduler.Instance;
        public TimeProvider TimeProvider { get; init; } = TimeProvider.System;
        public Func<double> JitterSource { get; init; } = Random.Shared.NextDouble;
    }

    public sealed class RetryPolicy
    {
        public bool Enabled { get; init; } = true;
        public int MaximumAttempts { get; init; } = 3;
        public TimeSpan MaximumElapsedTime { get; init; } = TimeSpan.FromMinutes(2);
        public TimeSpan BaseDelay { get; init; } = TimeSpan.FromMilliseconds(500);
        public TimeSpan MaximumDelay { get; init; } = TimeSpan.FromSeconds(30);
        public bool UseJitter { get; init; } = true;

        internal void Validate()
        {
            if (MaximumAttempts < 1)
                throw new ArgumentOutOfRangeException(nameof(MaximumAttempts), "MaximumAttempts must be at least one.");
            if (MaximumElapsedTime < TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(MaximumElapsedTime));
            if (BaseDelay < TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(BaseDelay));
            if (MaximumDelay < TimeSpan.Zero || MaximumDelay < BaseDelay)
                throw new ArgumentOutOfRangeException(nameof(MaximumDelay), "MaximumDelay must be at least BaseDelay.");
        }
    }

    /// <summary>
    /// Optional controls for one logical request. Service-wide transport
    /// settings remain the default when these values are omitted.
    /// </summary>
    public sealed class RequestTransportOptions
    {
        public RetryPolicy? Retry { get; init; }
        public int? EstimatedInputTokens { get; init; }

        internal void Validate()
        {
            Retry?.Validate();
            if (EstimatedInputTokens < 0)
                throw new ArgumentOutOfRangeException(nameof(EstimatedInputTokens));
        }
    }

    public sealed class RetryAttempt
    {
        public int AttemptNumber { get; init; }
        public DateTimeOffset StartedAt { get; init; }
        public TimeSpan Duration { get; init; }
        public HttpStatusCode? StatusCode { get; init; }
        public string? ExceptionType { get; init; }
        public string? ExceptionMessage { get; init; }
        public TimeSpan? ServerDelay { get; init; }
        public TimeSpan? RetryDelay { get; init; }
        public bool WillRetry { get; init; }
    }

    public sealed class RetryContext
    {
        public required LLMProvider Provider { get; init; }
        public required string Endpoint { get; init; }
        public required string Model { get; init; }
        public required RetryAttempt Attempt { get; init; }
    }

    public interface IRetryObserver
    {
        ValueTask OnRetryAsync(RetryContext context, CancellationToken cancellationToken = default);
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
    /// request, including retries.
    /// </summary>
    public interface IRequestAdmissionPolicy
    {
        ValueTask<IDisposable?> AcquireAsync(
            RequestAdmissionContext context,
            CancellationToken cancellationToken = default);
    }

    public interface IRetryDelayScheduler
    {
        ValueTask DelayAsync(TimeSpan delay, CancellationToken cancellationToken = default);
    }

    public sealed class SystemRetryDelayScheduler : IRetryDelayScheduler
    {
        public static SystemRetryDelayScheduler Instance { get; } = new();
        private SystemRetryDelayScheduler() { }

        public async ValueTask DelayAsync(
            TimeSpan delay,
            CancellationToken cancellationToken = default)
        {
            if (delay > TimeSpan.Zero)
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
        }
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
        public IReadOnlyList<RetryAttempt> Attempts { get; init; } = Array.Empty<RetryAttempt>();
        public RateLimitSnapshot? RateLimits { get; init; }
    }
}
