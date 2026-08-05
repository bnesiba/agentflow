using System.Collections.Generic;
using System.Text.Json;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// Stable identifiers used for provider-bound native state.
    /// </summary>
    public static class ProviderIds
    {
        public const string Claude = "anthropic";
        public const string OpenAI = "openai";
        public const string Gemini = "google-gemini";
    }

    /// <summary>
    /// Lossless provider-native JSON associated with a portable content block.
    /// The native value is authoritative when replayed to the same provider.
    /// </summary>
    public sealed class ProviderNativeRepresentation
    {
        public string Provider { get; set; } = string.Empty;

        public JsonElement Value { get; set; }

        public static ProviderNativeRepresentation Create(
            string provider,
            JsonElement value)
        {
            return new ProviderNativeRepresentation
            {
                Provider = provider,
                Value = value.Clone()
            };
        }
    }

    /// <summary>
    /// Determines how response-level continuation state is used on a later request.
    /// </summary>
    public enum ContinuationMode
    {
        /// <summary>
        /// Replay the retained native items in a stateless request.
        /// </summary>
        StatelessReplay,

        /// <summary>
        /// Refer to provider-managed state by response identifier.
        /// </summary>
        ServerManaged
    }

    /// <summary>
    /// Provider-bound response state required to continue a conversation without
    /// flattening native output items into portable chat messages.
    /// </summary>
    public sealed class ProviderContinuationState
    {
        public string Provider { get; set; } = string.Empty;

        public ContinuationMode Mode { get; set; } = ContinuationMode.StatelessReplay;

        public string? ResponseId { get; set; }

        public List<JsonElement> NativeItems { get; set; } = new();
    }
}
