using System;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;

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
    }

    /// <summary>
    /// Factory for creating LLM service instances
    /// </summary>
    public static class LLMServiceFactory
    {
        /// <summary>
        /// Create an LLM service based on configuration
        /// </summary>
        public static ILLMService Create(LLMServiceConfig config)
        {
            if (string.IsNullOrEmpty(config.ApiKey))
            {
                throw new ArgumentException("API key is required", nameof(config.ApiKey));
            }

            return config.Provider switch
            {
                LLMProvider.OpenAI => new OpenAIService(config.ApiKey, config.BaseUrl),
                LLMProvider.Claude => new ClaudeService(
                    config.ApiKey, 
                    config.ApiVersion, 
                    config.BaseUrl),
                LLMProvider.Gemini => new GeminiService(config.ApiKey, config.BaseUrl),
                _ => throw new ArgumentException($"Unknown provider: {config.Provider}")
            };
        }

        /// <summary>
        /// Create an OpenAI service
        /// </summary>
        public static ILLMService CreateOpenAI(string apiKey, string? baseUrl = null)
        {
            return new OpenAIService(apiKey, baseUrl);
        }

        /// <summary>
        /// Create a Claude service
        /// </summary>
        public static ILLMService CreateClaude(
            string apiKey, 
            string? apiVersion = null, 
            string? baseUrl = null)
        {
            return new ClaudeService(apiKey, apiVersion, baseUrl);
        }

        /// <summary>
        /// Create a Gemini service
        /// </summary>
        public static ILLMService CreateGemini(string apiKey, string? baseUrl = null)
        {
            return new GeminiService(apiKey, baseUrl);
        }
    }
}
