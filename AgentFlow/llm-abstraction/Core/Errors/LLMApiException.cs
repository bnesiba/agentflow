using System;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using LLMAbstraction.Core.Transport;

namespace LLMAbstraction.Core.Errors
{
    public sealed class LLMApiException : HttpRequestException
    {
        public LLMApiException(
            LLMProvider provider,
            HttpStatusCode statusCode,
            string message,
            string? errorCode,
            string? errorType,
            string? parameter,
            string? requestId,
            string rawResponse,
            JsonElement? details = null,
            TransportMetadata? transport = null)
            : base(message, null, statusCode)
        {
            Provider = provider;
            ErrorCode = errorCode;
            ErrorType = errorType;
            Parameter = parameter;
            RequestId = requestId;
            RawResponse = rawResponse;
            Details = details?.Clone();
            Transport = transport;
            IsTransient = statusCode is HttpStatusCode.RequestTimeout or HttpStatusCode.Conflict or
                HttpStatusCode.TooManyRequests || (int)statusCode >= 500 ||
                errorType is "overloaded_error" or "rate_limit_error" or "RESOURCE_EXHAUSTED" or "UNAVAILABLE";
        }

        /// <summary>True when stream content was delivered before this failure.</summary>
        public bool HasPartialOutput { get; internal set; }
        public bool HasToolCalls { get; internal set; }
        public LLMProvider Provider { get; }
        public string? ErrorCode { get; }
        public string? ErrorType { get; }
        public string? Parameter { get; }
        public string? RequestId { get; }
        public bool IsTransient { get; }
        public string RawResponse { get; }
        public JsonElement? Details { get; }
        public TransportMetadata? Transport { get; }
    }
}
