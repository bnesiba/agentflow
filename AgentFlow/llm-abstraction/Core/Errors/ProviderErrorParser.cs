using System;
using System.Linq;
using System.Net.Http;
using System.Text.Json;

namespace LLMAbstraction.Core.Errors
{
    internal static class ProviderErrorParser
    {
        public static LLMApiException Create(
            LLMProvider provider,
            HttpResponseMessage response,
            string rawResponse)
        {
            string? message = null;
            string? code = null;
            string? type = null;
            string? parameter = null;
            JsonElement? details = null;

            try
            {
                using var document = JsonDocument.Parse(rawResponse);
                var root = document.RootElement;
                if (root.TryGetProperty("error", out var error) && error.ValueKind == JsonValueKind.Object)
                {
                    message = GetString(error, "message");
                    type = GetString(error, "type") ?? GetString(error, "status");
                    parameter = GetString(error, "param");
                    code = GetFlexibleString(error, "code") ?? GetString(error, "status");
                    if (error.TryGetProperty("details", out var errorDetails))
                        details = errorDetails.Clone();
                }

                type ??= GetString(root, "type");
                message ??= GetString(root, "message");
            }
            catch (JsonException)
            {
                // Preserve non-JSON provider and proxy responses as raw text.
            }

            message ??= $"{provider} API request failed with status {(int)response.StatusCode} ({response.StatusCode}).";
            var requestId = GetHeader(response, "request-id") ??
                GetHeader(response, "x-request-id") ??
                GetHeader(response, "x-goog-request-id");

            return new LLMApiException(
                provider,
                response.StatusCode,
                message,
                code,
                type,
                parameter,
                requestId,
                GetRetryAfter(response),
                rawResponse,
                details);
        }

        private static string? GetString(JsonElement value, string property)
        {
            return value.TryGetProperty(property, out var result) && result.ValueKind == JsonValueKind.String
                ? result.GetString()
                : null;
        }

        private static string? GetFlexibleString(JsonElement value, string property)
        {
            if (!value.TryGetProperty(property, out var result))
                return null;
            return result.ValueKind == JsonValueKind.String ? result.GetString() : result.GetRawText();
        }

        private static string? GetHeader(HttpResponseMessage response, string name)
        {
            return response.Headers.TryGetValues(name, out var values)
                ? values.FirstOrDefault()
                : null;
        }

        private static TimeSpan? GetRetryAfter(HttpResponseMessage response)
        {
            if (response.Headers.RetryAfter?.Delta != null)
                return response.Headers.RetryAfter.Delta;
            if (response.Headers.RetryAfter?.Date != null)
            {
                var difference = response.Headers.RetryAfter.Date.Value - DateTimeOffset.UtcNow;
                return difference > TimeSpan.Zero ? difference : TimeSpan.Zero;
            }
            return null;
        }
    }
}
