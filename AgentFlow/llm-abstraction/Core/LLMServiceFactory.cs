using System;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Core.Transport;

namespace LLMAbstraction.Core
{
    /// <summary>
    /// Provider types supported by the abstraction layer
    /// </summary>
    public enum LLMProvider
    {
        OpenAI,
        Claude,
        Gemini
    }

    /// <summary>
    /// Configuration for LLM service
    /// </summary>
    public class LLMServiceConfig
    {
        public LLMProvider Provider { get; set; }
        public string ApiKey { get; set; } = string.Empty;
        public string? BaseUrl { get; set; }
        public string? ApiVersion { get; set; }  // For Claude
        public LLMTransportOptions? Transport { get; set; }
    }

    /// <summary>
    /// Factory for creating LLM service instances
    /// </summary>
    public static class LLMServiceFactory
    {
        /// <summary>
        /// Create an LLM service based on configuration
        /// </summary>
        public static ILLMServiceWithTokenCounting Create(LLMServiceConfig config)
        {
            if (string.IsNullOrEmpty(config.ApiKey))
            {
                throw new ArgumentException("API key is required", nameof(config.ApiKey));
            }

            return config.Provider switch
            {
                LLMProvider.OpenAI => new OpenAIService(config.ApiKey, config.BaseUrl, config.Transport),
                LLMProvider.Claude => new ClaudeService(
                    config.ApiKey, 
                    config.ApiVersion, 
                    config.BaseUrl,
                    config.Transport),
                LLMProvider.Gemini => new GeminiService(
                    config.ApiKey,
                    config.BaseUrl,
                    transportOptions: config.Transport),
                _ => throw new ArgumentException($"Unknown provider: {config.Provider}")
            };
        }

        /// <summary>
        /// Create an OpenAI service
        /// </summary>
        public static ILLMServiceWithTokenCounting CreateOpenAI(
            string apiKey,
            string? baseUrl = null,
            LLMTransportOptions? transport = null)
        {
            return new OpenAIService(apiKey, baseUrl, transport);
        }

        /// <summary>
        /// Create a Claude service
        /// </summary>
        public static ILLMServiceWithTokenCounting CreateClaude(
            string apiKey, 
            string? apiVersion = null, 
            string? baseUrl = null,
            LLMTransportOptions? transport = null)
        {
            return new ClaudeService(apiKey, apiVersion, baseUrl, transport);
        }

        /// <summary>
        /// Create a Gemini service
        /// </summary>
        public static ILLMServiceWithTokenCounting CreateGemini(
            string apiKey,
            string? baseUrl = null,
            LLMTransportOptions? transport = null)
        {
            return new GeminiService(apiKey, baseUrl, transportOptions: transport);
        }

        /// <summary>Create a Gemini explicit prompt-cache resource service.</summary>
        public static IPromptCacheService CreateGeminiPromptCache(
            string apiKey,
            string? baseUrl = null,
            LLMTransportOptions? transport = null)
        {
            return new GeminiService(apiKey, baseUrl, transportOptions: transport);
        }
    }
}
