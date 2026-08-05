using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace LLMAbstraction.Core.Models
{
    internal static class ProviderOptionMerger
    {
        public static Dictionary<string, JsonElement>? ConvertAdditionalFields(
            Dictionary<string, object>? options,
            IReadOnlySet<string> protectedFields,
            IReadOnlySet<string>? separatelyHandledFields = null)
        {
            if (options == null || options.Count == 0)
                return null;

            var result = new Dictionary<string, JsonElement>(StringComparer.Ordinal);
            foreach (var option in options)
            {
                if (separatelyHandledFields?.Contains(option.Key) == true)
                    continue;

                if (protectedFields.Contains(option.Key))
                {
                    throw new ArgumentException(
                        $"Provider option '{option.Key}' conflicts with a unified request field and cannot override it.");
                }

                result[option.Key] = option.Value is JsonElement json
                    ? json.Clone()
                    : JsonSerializer.SerializeToElement(option.Value, option.Value?.GetType() ?? typeof(object));
            }

            return result.Count == 0 ? null : result;
        }
    }
}
