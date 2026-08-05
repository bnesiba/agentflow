using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;

namespace LLMAbstraction.Core.Transport
{
    internal static partial class RateLimitHeaderParser
    {
        public static RateLimitSnapshot? Parse(HttpResponseMessage response, DateTimeOffset now)
        {
            var headers = response.Headers
                .Concat(response.Content.Headers)
                .GroupBy(header => header.Key, StringComparer.OrdinalIgnoreCase)
                .ToDictionary(
                    group => group.Key,
                    group => group.SelectMany(header => header.Value).ToArray(),
                    StringComparer.OrdinalIgnoreCase);

            var requests = ParseWindow(headers, now,
                ("anthropic-ratelimit-requests-limit", "anthropic-ratelimit-requests-remaining", "anthropic-ratelimit-requests-reset"),
                ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests"));
            var tokens = ParseWindow(headers, now,
                ("anthropic-ratelimit-tokens-limit", "anthropic-ratelimit-tokens-remaining", "anthropic-ratelimit-tokens-reset"),
                ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"));
            var input = ParseWindow(headers, now,
                ("anthropic-ratelimit-input-tokens-limit", "anthropic-ratelimit-input-tokens-remaining", "anthropic-ratelimit-input-tokens-reset"));
            var output = ParseWindow(headers, now,
                ("anthropic-ratelimit-output-tokens-limit", "anthropic-ratelimit-output-tokens-remaining", "anthropic-ratelimit-output-tokens-reset"));
            var hasKnown = requests != null || tokens != null || input != null || output != null;
            var rawRateHeaders = headers
                .Where(header =>
                    header.Key.Contains("ratelimit", StringComparison.OrdinalIgnoreCase) ||
                    header.Key.Equals("retry-after", StringComparison.OrdinalIgnoreCase))
                .ToDictionary(header => header.Key, header => header.Value, StringComparer.OrdinalIgnoreCase);

            if (!hasKnown && rawRateHeaders.Count == 0)
                return null;
            return new RateLimitSnapshot
            {
                Requests = requests,
                Tokens = tokens,
                InputTokens = input,
                OutputTokens = output,
                RawHeaders = rawRateHeaders
            };
        }

        public static string? GetRequestId(HttpResponseMessage response) =>
            Get(response, "request-id") ??
            Get(response, "x-request-id") ??
            Get(response, "x-goog-request-id");

        public static TimeSpan? GetServerDelay(HttpResponseMessage response, DateTimeOffset now)
        {
            if (response.Headers.RetryAfter?.Delta is TimeSpan delta)
                return NonNegative(delta);
            if (response.Headers.RetryAfter?.Date is DateTimeOffset date)
                return NonNegative(date - now);

            foreach (var name in new[]
                     {
                         "anthropic-ratelimit-requests-reset",
                         "x-ratelimit-reset-requests",
                         "x-ratelimit-reset-tokens"
                     })
            {
                if (Get(response, name) is string value && TryParseReset(value, now, out var reset))
                    return reset;
            }
            return null;
        }

        private static RateLimitWindow? ParseWindow(
            IReadOnlyDictionary<string, string[]> headers,
            DateTimeOffset now,
            params (string Limit, string Remaining, string Reset)[] names)
        {
            foreach (var name in names)
            {
                var limit = ParseLong(Get(headers, name.Limit));
                var remaining = ParseLong(Get(headers, name.Remaining));
                var resetValue = Get(headers, name.Reset);
                if (limit == null && remaining == null && resetValue == null)
                    continue;

                DateTimeOffset? resetsAt = null;
                TimeSpan? resetsAfter = null;
                if (resetValue != null && TryParseReset(resetValue, now, out var reset))
                {
                    resetsAfter = reset;
                    resetsAt = now + reset;
                }
                return new RateLimitWindow
                {
                    Limit = limit,
                    Remaining = remaining,
                    ResetsAt = resetsAt,
                    ResetsAfter = resetsAfter
                };
            }
            return null;
        }

        private static bool TryParseReset(string value, DateTimeOffset now, out TimeSpan result)
        {
            if (DateTimeOffset.TryParse(value, CultureInfo.InvariantCulture,
                    DateTimeStyles.AssumeUniversal, out var timestamp))
            {
                result = NonNegative(timestamp - now);
                return true;
            }
            if (long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var integer))
            {
                // Provider reset headers commonly use epoch seconds. Small values
                // are treated as a relative duration.
                result = integer > 1_000_000_000
                    ? NonNegative(DateTimeOffset.FromUnixTimeSeconds(integer) - now)
                    : TimeSpan.FromSeconds(Math.Max(0, integer));
                return true;
            }

            var matches = DurationPartRegex().Matches(value);
            if (matches.Count == 0)
            {
                result = default;
                return false;
            }
            double milliseconds = 0;
            foreach (Match match in matches)
            {
                var amount = double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
                milliseconds += match.Groups[2].Value switch
                {
                    "ms" => amount,
                    "s" => amount * 1000,
                    "m" => amount * 60_000,
                    "h" => amount * 3_600_000,
                    _ => 0
                };
            }
            result = TimeSpan.FromMilliseconds(milliseconds);
            return true;
        }

        private static long? ParseLong(string? value) =>
            long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
                ? parsed
                : null;

        private static string? Get(HttpResponseMessage response, string name) =>
            response.Headers.TryGetValues(name, out var values)
                ? values.FirstOrDefault()
                : response.Content.Headers.TryGetValues(name, out values)
                    ? values.FirstOrDefault()
                    : null;

        private static string? Get(IReadOnlyDictionary<string, string[]> headers, string name) =>
            headers.TryGetValue(name, out var values) ? values.FirstOrDefault() : null;

        private static TimeSpan NonNegative(TimeSpan value) =>
            value > TimeSpan.Zero ? value : TimeSpan.Zero;

        [GeneratedRegex(@"(\d+(?:\.\d+)?)(ms|s|m|h)", RegexOptions.IgnoreCase)]
        private static partial Regex DurationPartRegex();
    }
}
