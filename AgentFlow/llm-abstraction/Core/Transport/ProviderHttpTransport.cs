using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Errors;

namespace LLMAbstraction.Core.Transport
{
    internal sealed class TransportResponse : IDisposable
    {
        public required HttpResponseMessage Response { get; init; }
        public required TransportMetadata Metadata { get; init; }
        public IDisposable? AdmissionLease { get; init; }
        public void Dispose()
        {
            Response.Dispose();
            AdmissionLease?.Dispose();
        }
    }

    internal sealed class ProviderHttpTransport
    {
        private readonly HttpClient _client;
        private readonly LLMProvider _provider;
        private readonly LLMTransportOptions _options;

        public ProviderHttpTransport(
            HttpClient client,
            LLMProvider provider,
            LLMTransportOptions? options = null)
        {
            _client = client ?? throw new ArgumentNullException(nameof(client));
            _provider = provider;
            _options = options ?? new LLMTransportOptions();
            _options.Retry.Validate();
            if (_options.JitterSource == null)
                throw new ArgumentNullException(nameof(options), "JitterSource cannot be null.");
        }

        public async Task<TransportResponse> SendAsync(
            Func<HttpRequestMessage> requestFactory,
            string endpoint,
            string model,
            bool isStreaming,
            CancellationToken cancellationToken,
            RequestTransportOptions? requestOptions = null)
        {
            ArgumentNullException.ThrowIfNull(requestFactory);
            requestOptions?.Validate();
            var retryPolicy = requestOptions?.Retry ?? _options.Retry;
            var lease = _options.AdmissionPolicy == null
                ? null
                : await _options.AdmissionPolicy.AcquireAsync(
                    new RequestAdmissionContext
                    {
                        Provider = _provider,
                        Endpoint = endpoint,
                        Model = model,
                        IsStreaming = isStreaming,
                        EstimatedInputTokens = requestOptions?.EstimatedInputTokens
                    },
                    cancellationToken).ConfigureAwait(false);

            var attempts = new List<RetryAttempt>();
            var operationStart = _options.TimeProvider.GetTimestamp();
            var ownershipTransferred = false;

            try
            {
                for (var attemptNumber = 1; ; attemptNumber++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var attemptStartTimestamp = _options.TimeProvider.GetTimestamp();
                    var startedAt = _options.TimeProvider.GetUtcNow();
                    HttpResponseMessage response;
                    try
                    {
                        using var request = requestFactory();
                        response = await _client.SendAsync(
                            request,
                            isStreaming
                                ? HttpCompletionOption.ResponseHeadersRead
                                : HttpCompletionOption.ResponseContentRead,
                            cancellationToken).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
                    {
                        var exception = new TimeoutException("The provider request timed out.");
                        if (!CanRetry(retryPolicy, attemptNumber, operationStart, null, out var retryDelay))
                        {
                            attempts.Add(CreateAttempt(attemptNumber, startedAt, attemptStartTimestamp,
                                null, exception, null, null, false));
                            throw new LLMTransportException(
                                _provider,
                                "The provider request timed out and the retry policy was exhausted.",
                                true,
                                new TransportMetadata { Attempts = attempts.ToArray() },
                                exception);
                        }

                        var attempt = CreateAttempt(attemptNumber, startedAt, attemptStartTimestamp,
                            null, exception, null, retryDelay, true);
                        attempts.Add(attempt);
                        await NotifyAndDelayAsync(endpoint, model, attempt, retryDelay, cancellationToken)
                            .ConfigureAwait(false);
                        continue;
                    }
                    catch (HttpRequestException exception)
                    {
                        var transient = exception.StatusCode == null || IsTransient(exception.StatusCode.Value);
                        if (!transient ||
                            !CanRetry(retryPolicy, attemptNumber, operationStart, null, out var retryDelay))
                        {
                            attempts.Add(CreateAttempt(attemptNumber, startedAt, attemptStartTimestamp,
                                exception.StatusCode, exception, null, null, false));
                            throw new LLMTransportException(
                                _provider,
                                transient
                                    ? "The provider request failed and the retry policy was exhausted."
                                    : "The provider request failed with a non-transient transport error.",
                                transient,
                                new TransportMetadata { Attempts = attempts.ToArray() },
                                exception);
                        }

                        var attempt = CreateAttempt(attemptNumber, startedAt, attemptStartTimestamp,
                            exception.StatusCode, exception, null, retryDelay, true);
                        attempts.Add(attempt);
                        await NotifyAndDelayAsync(endpoint, model, attempt, retryDelay, cancellationToken)
                            .ConfigureAwait(false);
                        continue;
                    }

                    try
                    {
                        var serverDelay = RateLimitHeaderParser.GetServerDelay(
                            response,
                            _options.TimeProvider.GetUtcNow());
                        var retryDelay = TimeSpan.Zero;
                        var retry = IsTransient(response.StatusCode) &&
                            CanRetry(retryPolicy, attemptNumber, operationStart, serverDelay, out retryDelay);
                        var attempt = CreateAttempt(
                            attemptNumber,
                            startedAt,
                            attemptStartTimestamp,
                            response.StatusCode,
                            null,
                            serverDelay,
                            retry ? retryDelay : null,
                            retry);
                        attempts.Add(attempt);

                        if (!retry)
                        {
                            ownershipTransferred = true;
                            return new TransportResponse
                            {
                                Response = response,
                                AdmissionLease = lease,
                                Metadata = new TransportMetadata
                                {
                                    RequestId = RateLimitHeaderParser.GetRequestId(response),
                                    Attempts = attempts.ToArray(),
                                    RateLimits = RateLimitHeaderParser.Parse(
                                        response,
                                        _options.TimeProvider.GetUtcNow())
                                }
                            };
                        }

                        response.Dispose();
                        await NotifyAndDelayAsync(endpoint, model, attempt, retryDelay, cancellationToken)
                            .ConfigureAwait(false);
                    }
                    catch
                    {
                        response.Dispose();
                        throw;
                    }
                }
            }
            finally
            {
                if (!ownershipTransferred)
                    lease?.Dispose();
            }
        }

        private bool CanRetry(
            RetryPolicy policy,
            int attemptNumber,
            long operationStart,
            TimeSpan? serverDelay,
            out TimeSpan delay)
        {
            delay = TimeSpan.Zero;
            if (!policy.Enabled || attemptNumber >= policy.MaximumAttempts)
                return false;

            var exponent = Math.Pow(2, attemptNumber - 1);
            var clientMilliseconds = Math.Min(
                policy.MaximumDelay.TotalMilliseconds,
                policy.BaseDelay.TotalMilliseconds * exponent);
            if (policy.UseJitter && clientMilliseconds > 0)
                clientMilliseconds *= 0.5 + Math.Clamp(_options.JitterSource(), 0, 1) * 0.5;
            delay = TimeSpan.FromMilliseconds(clientMilliseconds);
            if (serverDelay > delay)
                delay = serverDelay.Value;

            var elapsed = _options.TimeProvider.GetElapsedTime(operationStart);
            return elapsed + delay <= policy.MaximumElapsedTime;
        }

        private async Task NotifyAndDelayAsync(
            string endpoint,
            string model,
            RetryAttempt attempt,
            TimeSpan delay,
            CancellationToken cancellationToken)
        {
            if (_options.RetryObserver != null)
            {
                await _options.RetryObserver.OnRetryAsync(
                    new RetryContext
                    {
                        Provider = _provider,
                        Endpoint = endpoint,
                        Model = model,
                        Attempt = attempt
                    },
                    cancellationToken).ConfigureAwait(false);
            }
            await _options.DelayScheduler.DelayAsync(delay, cancellationToken).ConfigureAwait(false);
        }

        private RetryAttempt CreateAttempt(
            int number,
            DateTimeOffset startedAt,
            long startTimestamp,
            HttpStatusCode? statusCode,
            Exception? exception,
            TimeSpan? serverDelay,
            TimeSpan? retryDelay,
            bool willRetry) => new()
        {
            AttemptNumber = number,
            StartedAt = startedAt,
            Duration = _options.TimeProvider.GetElapsedTime(startTimestamp),
            StatusCode = statusCode,
            ExceptionType = exception?.GetType().FullName,
            ExceptionMessage = exception?.Message,
            ServerDelay = serverDelay,
            RetryDelay = retryDelay,
            WillRetry = willRetry
        };

        private static bool IsTransient(HttpStatusCode statusCode) =>
            statusCode is HttpStatusCode.RequestTimeout or HttpStatusCode.Conflict or
                HttpStatusCode.TooManyRequests || (int)statusCode >= 500;
    }
}
