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

            var startedAt = _options.TimeProvider.GetUtcNow();
            var started = _options.TimeProvider.GetTimestamp();
            var transferred = false;
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                using var request = requestFactory();
                HttpResponseMessage response;
                try
                {
                    response = await _client.SendAsync(request,
                        isStreaming ? HttpCompletionOption.ResponseHeadersRead : HttpCompletionOption.ResponseContentRead,
                        cancellationToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException exception) when (!cancellationToken.IsCancellationRequested)
                {
                    throw Failure("The provider request timed out.", exception);
                }
                catch (HttpRequestException exception)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    throw Failure("The provider connection failed.", exception);
                }
                catch (System.IO.IOException exception)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    throw Failure("The provider connection failed.", exception);
                }

                try
                {
                    var metadata = new TransportMetadata
                    {
                        RequestId = RateLimitHeaderParser.GetRequestId(response),
                        RateLimits = RateLimitHeaderParser.Parse(response, _options.TimeProvider.GetUtcNow()),
                        Attempts = new[] { CreateAttempt(1, startedAt, started, response.StatusCode, null) }
                    };
                    transferred = true;
                    return new TransportResponse { Response = response, Metadata = metadata, AdmissionLease = lease };
                }
                catch { response.Dispose(); throw; }
            }
            finally
            {
                if (!transferred) lease?.Dispose();
            }

            LLMTransportException Failure(string message, Exception exception) => new(
                _provider, message, true,
                new TransportMetadata { Attempts = new[] { CreateAttempt(1, startedAt, started, null, exception) } },
                exception);
        }

        private TransportAttempt CreateAttempt(
            int number,
            DateTimeOffset startedAt,
            long startTimestamp,
            HttpStatusCode? statusCode,
            Exception? exception) => new()
        {
            AttemptNumber = number,
            StartedAt = startedAt,
            Duration = _options.TimeProvider.GetElapsedTime(startTimestamp),
            StatusCode = statusCode,
            ExceptionType = exception?.GetType().FullName,
            ExceptionMessage = exception?.Message
        };

    }
}
