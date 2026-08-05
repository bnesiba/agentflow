using System;
using System.Collections.Generic;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Core.Validation
{
    /// <summary>
    /// The concrete provider API surface used to serialize a request. Capabilities
    /// may differ for the same model when a provider exposes multiple surfaces.
    /// </summary>
    public enum LLMApiSurface
    {
        AnthropicMessages,
        OpenAIResponses,
        GeminiInteractions,
        GeminiGenerateContent
    }

    public enum CapabilitySupport
    {
        Unknown,
        Unsupported,
        Supported
    }

    public enum ModelRecognition
    {
        Unknown,
        Known
    }

    /// <summary>
    /// Model- and endpoint-specific capabilities used by request validation.
    /// Unknown models intentionally remain usable for basic text requests, but
    /// advanced settings are not optimistically assumed to work.
    /// </summary>
    public sealed class ModelCapabilityProfile
    {
        public required LLMProvider Provider { get; init; }
        public required LLMApiSurface Surface { get; init; }
        public required string Model { get; init; }
        public required string Family { get; init; }
        public ModelRecognition Recognition { get; init; }
        public CapabilitySupport StructuredOutput { get; init; }
        public CapabilitySupport JsonObjectOutput { get; init; }
        public CapabilitySupport Reasoning { get; init; }
        public CapabilitySupport ThinkingBudget { get; init; }
        public CapabilitySupport AnthropicAdaptiveThinking { get; init; }
        public CapabilitySupport AnthropicManualThinking { get; init; }
        public CapabilitySupport OpenAIProMode { get; init; }
        public CapabilitySupport OpenAIReasoningContext { get; init; }
        public CapabilitySupport SamplingControls { get; init; } = CapabilitySupport.Supported;
        public double MinimumTemperature { get; init; }
        public double MaximumTemperature { get; init; }
        public IReadOnlySet<ReasoningEffort> ReasoningEfforts { get; init; } =
            new HashSet<ReasoningEffort>();
    }

    public static class ModelCapabilityRegistry
    {
        private static readonly IReadOnlySet<ReasoningEffort> ClaudeModernEfforts = Set(
            ReasoningEffort.Low,
            ReasoningEffort.Medium,
            ReasoningEffort.High,
            ReasoningEffort.XHigh,
            ReasoningEffort.Max);

        private static readonly IReadOnlySet<ReasoningEffort> OpenAI56Efforts = Set(
            ReasoningEffort.None,
            ReasoningEffort.Low,
            ReasoningEffort.Medium,
            ReasoningEffort.High,
            ReasoningEffort.XHigh,
            ReasoningEffort.Max);

        private static readonly IReadOnlySet<ReasoningEffort> GeminiAllLevels = Set(
            ReasoningEffort.Minimal,
            ReasoningEffort.Low,
            ReasoningEffort.Medium,
            ReasoningEffort.High);

        public static ModelCapabilityProfile Resolve(
            LLMProvider provider,
            string model,
            LLMApiSurface? surface = null)
        {
            var resolvedSurface = surface ?? DefaultSurface(provider);
            EnsureSurfaceMatchesProvider(provider, resolvedSurface);
            var normalized = model?.Trim() ?? string.Empty;

            return provider switch
            {
                LLMProvider.Claude => ResolveClaude(normalized, resolvedSurface),
                LLMProvider.OpenAI => ResolveOpenAI(normalized, resolvedSurface),
                LLMProvider.Gemini => ResolveGemini(normalized, resolvedSurface),
                _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, null)
            };
        }

        public static LLMApiSurface DefaultSurface(LLMProvider provider) => provider switch
        {
            LLMProvider.Claude => LLMApiSurface.AnthropicMessages,
            LLMProvider.OpenAI => LLMApiSurface.OpenAIResponses,
            LLMProvider.Gemini => LLMApiSurface.GeminiInteractions,
            _ => throw new ArgumentOutOfRangeException(nameof(provider), provider, null)
        };

        private static ModelCapabilityProfile ResolveClaude(string model, LLMApiSurface surface)
        {
            if (StartsWith(model, "claude-sonnet-5") || StartsWith(model, "claude-opus-5"))
            {
                return Known(LLMProvider.Claude, surface, model, "claude-5",
                    structured: CapabilitySupport.Supported,
                    jsonObject: CapabilitySupport.Unsupported,
                    reasoning: CapabilitySupport.Supported,
                    efforts: ClaudeModernEfforts,
                    adaptive: CapabilitySupport.Supported,
                    manual: CapabilitySupport.Unsupported,
                    sampling: CapabilitySupport.Unsupported,
                    maxTemperature: 1);
            }

            if (StartsWith(model, "claude-sonnet-4-6") || StartsWith(model, "claude-opus-4-6"))
            {
                return Known(LLMProvider.Claude, surface, model, "claude-4.6",
                    structured: CapabilitySupport.Supported,
                    jsonObject: CapabilitySupport.Unsupported,
                    reasoning: CapabilitySupport.Supported,
                    efforts: ClaudeModernEfforts,
                    adaptive: CapabilitySupport.Supported,
                    manual: CapabilitySupport.Supported,
                    sampling: CapabilitySupport.Supported,
                    maxTemperature: 1);
            }

            if (StartsWith(model, "claude-opus-4-8") || StartsWith(model, "claude-opus-4-7"))
            {
                return Known(LLMProvider.Claude, surface, model, "claude-opus-4.7+",
                    structured: CapabilitySupport.Supported,
                    jsonObject: CapabilitySupport.Unsupported,
                    reasoning: CapabilitySupport.Supported,
                    efforts: ClaudeModernEfforts,
                    adaptive: CapabilitySupport.Supported,
                    manual: CapabilitySupport.Unsupported,
                    sampling: CapabilitySupport.Supported,
                    maxTemperature: 1);
            }

            return Unknown(LLMProvider.Claude, surface, model, maxTemperature: 1);
        }

        private static ModelCapabilityProfile ResolveOpenAI(string model, LLMApiSurface surface)
        {
            if (StartsWith(model, "gpt-5.6"))
            {
                return Known(LLMProvider.OpenAI, surface, model, "gpt-5.6",
                    structured: CapabilitySupport.Supported,
                    reasoning: CapabilitySupport.Supported,
                    efforts: OpenAI56Efforts,
                    pro: CapabilitySupport.Supported,
                    context: CapabilitySupport.Supported,
                    maxTemperature: 2);
            }

            if (StartsWith(model, "gpt-5.4-pro"))
            {
                return Known(LLMProvider.OpenAI, surface, model, "gpt-5.4-pro",
                    structured: CapabilitySupport.Unsupported,
                    reasoning: CapabilitySupport.Supported,
                    efforts: Set(ReasoningEffort.Medium, ReasoningEffort.High, ReasoningEffort.XHigh),
                    maxTemperature: 2);
            }

            if (StartsWith(model, "gpt-5.5") || StartsWith(model, "gpt-5.4"))
            {
                return Known(LLMProvider.OpenAI, surface, model, "gpt-5.4+",
                    structured: CapabilitySupport.Supported,
                    reasoning: CapabilitySupport.Supported,
                    efforts: Set(ReasoningEffort.None, ReasoningEffort.Low, ReasoningEffort.Medium,
                        ReasoningEffort.High, ReasoningEffort.XHigh),
                    maxTemperature: 2);
            }

            if (StartsWith(model, "gpt-4.1") || StartsWith(model, "gpt-4o"))
            {
                return Known(LLMProvider.OpenAI, surface, model, "gpt-4-class",
                    structured: CapabilitySupport.Supported,
                    reasoning: CapabilitySupport.Unsupported,
                    efforts: Set(),
                    maxTemperature: 2);
            }

            return Unknown(LLMProvider.OpenAI, surface, model, maxTemperature: 2);
        }

        private static ModelCapabilityProfile ResolveGemini(string model, LLMApiSurface surface)
        {
            if (StartsWith(model, "gemini-3.5-flash-lite"))
                return GeminiKnown(model, surface, "gemini-3.5-flash-lite", GeminiAllLevels);

            if (StartsWith(model, "gemini-3.6-flash") ||
                StartsWith(model, "gemini-3.5-flash") ||
                StartsWith(model, "gemini-3-flash"))
            {
                return GeminiKnown(model, surface, "gemini-flash-3+", GeminiAllLevels);
            }

            if (StartsWith(model, "gemini-3.1-pro"))
            {
                return GeminiKnown(model, surface, "gemini-3.1-pro",
                    Set(ReasoningEffort.Low, ReasoningEffort.Medium, ReasoningEffort.High));
            }

            if (StartsWith(model, "gemini-3-pro"))
            {
                return GeminiKnown(model, surface, "gemini-3-pro",
                    Set(ReasoningEffort.Low, ReasoningEffort.High));
            }

            if (StartsWith(model, "gemini-2.5-pro") || StartsWith(model, "gemini-2.5-flash"))
            {
                var profile = GeminiKnown(model, surface, "gemini-2.5",
                    surface == LLMApiSurface.GeminiInteractions
                        ? Set(ReasoningEffort.Low, ReasoningEffort.Medium, ReasoningEffort.High)
                        : Set());
                return CopyWithBudget(profile,
                    surface == LLMApiSurface.GeminiGenerateContent
                        ? CapabilitySupport.Supported
                        : CapabilitySupport.Unsupported);
            }

            return Unknown(LLMProvider.Gemini, surface, model, maxTemperature: 2);
        }

        private static ModelCapabilityProfile GeminiKnown(
            string model,
            LLMApiSurface surface,
            string family,
            IReadOnlySet<ReasoningEffort> efforts) =>
            Known(LLMProvider.Gemini, surface, model, family,
                structured: CapabilitySupport.Supported,
                reasoning: CapabilitySupport.Supported,
                efforts: efforts,
                maxTemperature: 2);

        private static ModelCapabilityProfile Known(
            LLMProvider provider,
            LLMApiSurface surface,
            string model,
            string family,
            CapabilitySupport structured,
            CapabilitySupport reasoning,
            IReadOnlySet<ReasoningEffort> efforts,
            CapabilitySupport jsonObject = CapabilitySupport.Supported,
            CapabilitySupport adaptive = CapabilitySupport.Unsupported,
            CapabilitySupport manual = CapabilitySupport.Unsupported,
            CapabilitySupport pro = CapabilitySupport.Unsupported,
            CapabilitySupport context = CapabilitySupport.Unsupported,
            CapabilitySupport sampling = CapabilitySupport.Supported,
            double maxTemperature = 2) => new()
        {
            Provider = provider,
            Surface = surface,
            Model = model,
            Family = family,
            Recognition = ModelRecognition.Known,
            StructuredOutput = structured,
            JsonObjectOutput = jsonObject,
            Reasoning = reasoning,
            ThinkingBudget = CapabilitySupport.Unsupported,
            AnthropicAdaptiveThinking = adaptive,
            AnthropicManualThinking = manual,
            OpenAIProMode = pro,
            OpenAIReasoningContext = context,
            SamplingControls = sampling,
            MinimumTemperature = 0,
            MaximumTemperature = maxTemperature,
            ReasoningEfforts = efforts
        };

        private static ModelCapabilityProfile Unknown(
            LLMProvider provider,
            LLMApiSurface surface,
            string model,
            double maxTemperature) => new()
        {
            Provider = provider,
            Surface = surface,
            Model = model,
            Family = "unknown",
            Recognition = ModelRecognition.Unknown,
            StructuredOutput = CapabilitySupport.Unknown,
            JsonObjectOutput = CapabilitySupport.Unknown,
            Reasoning = CapabilitySupport.Unknown,
            ThinkingBudget = CapabilitySupport.Unknown,
            AnthropicAdaptiveThinking = CapabilitySupport.Unknown,
            AnthropicManualThinking = CapabilitySupport.Unknown,
            OpenAIProMode = CapabilitySupport.Unknown,
            OpenAIReasoningContext = CapabilitySupport.Unknown,
            SamplingControls = CapabilitySupport.Unknown,
            MinimumTemperature = 0,
            MaximumTemperature = maxTemperature
        };

        private static ModelCapabilityProfile CopyWithBudget(
            ModelCapabilityProfile source,
            CapabilitySupport budget) => new()
        {
            Provider = source.Provider,
            Surface = source.Surface,
            Model = source.Model,
            Family = source.Family,
            Recognition = source.Recognition,
            StructuredOutput = source.StructuredOutput,
            JsonObjectOutput = source.JsonObjectOutput,
            Reasoning = source.Reasoning,
            ThinkingBudget = budget,
            AnthropicAdaptiveThinking = source.AnthropicAdaptiveThinking,
            AnthropicManualThinking = source.AnthropicManualThinking,
            OpenAIProMode = source.OpenAIProMode,
            OpenAIReasoningContext = source.OpenAIReasoningContext,
            SamplingControls = source.SamplingControls,
            MinimumTemperature = source.MinimumTemperature,
            MaximumTemperature = source.MaximumTemperature,
            ReasoningEfforts = source.ReasoningEfforts
        };

        private static IReadOnlySet<ReasoningEffort> Set(params ReasoningEffort[] efforts) =>
            new HashSet<ReasoningEffort>(efforts);

        private static bool StartsWith(string model, string prefix) =>
            model.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);

        private static void EnsureSurfaceMatchesProvider(LLMProvider provider, LLMApiSurface surface)
        {
            var matches = provider switch
            {
                LLMProvider.Claude => surface == LLMApiSurface.AnthropicMessages,
                LLMProvider.OpenAI => surface == LLMApiSurface.OpenAIResponses,
                LLMProvider.Gemini => surface is LLMApiSurface.GeminiInteractions or LLMApiSurface.GeminiGenerateContent,
                _ => false
            };
            if (!matches)
                throw new ArgumentException($"API surface {surface} does not belong to provider {provider}.", nameof(surface));
        }
    }
}
