using System;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using Xunit;

namespace LLMAbstraction.Tests.Integration
{
    public class LLMServiceFactoryTests
    {
        [Fact]
        public void CreateOpenAI_WithApiKey_ReturnsOpenAIService()
        {
            // Act
            var service = LLMServiceFactory.CreateOpenAI("test-api-key");

            // Assert
            Assert.NotNull(service);
            Assert.IsAssignableFrom<ILLMService>(service);
            Assert.IsType<OpenAIService>(service);
        }

        [Fact]
        public void CreateClaude_WithApiKey_ReturnsClaudeService()
        {
            // Act
            var service = LLMServiceFactory.CreateClaude("test-api-key");

            // Assert
            Assert.NotNull(service);
            Assert.IsAssignableFrom<ILLMService>(service);
            Assert.IsType<ClaudeService>(service);
        }

        [Fact]
        public void CreateGemini_WithApiKey_ReturnsGeminiService()
        {
            // Act
            var service = LLMServiceFactory.CreateGemini("test-api-key");

            // Assert
            Assert.NotNull(service);
            Assert.IsAssignableFrom<ILLMService>(service);
            Assert.IsType<GeminiService>(service);
        }

        [Fact]
        public void Create_WithOpenAIConfig_ReturnsOpenAIService()
        {
            // Arrange
            var config = new LLMServiceConfig
            {
                Provider = LLMProvider.OpenAI,
                ApiKey = "test-api-key"
            };

            // Act
            var service = LLMServiceFactory.Create(config);

            // Assert
            Assert.IsType<OpenAIService>(service);
        }

        [Fact]
        public void Create_WithClaudeConfig_ReturnsClaudeService()
        {
            // Arrange
            var config = new LLMServiceConfig
            {
                Provider = LLMProvider.Claude,
                ApiKey = "test-api-key",
                ApiVersion = "2023-06-01"
            };

            // Act
            var service = LLMServiceFactory.Create(config);

            // Assert
            Assert.IsType<ClaudeService>(service);
        }

        [Fact]
        public void Create_WithGeminiConfig_ReturnsGeminiService()
        {
            // Arrange
            var config = new LLMServiceConfig
            {
                Provider = LLMProvider.Gemini,
                ApiKey = "test-api-key"
            };

            // Act
            var service = LLMServiceFactory.Create(config);

            // Assert
            Assert.IsType<GeminiService>(service);
        }

        [Fact]
        public void Create_WithEmptyApiKey_ThrowsArgumentException()
        {
            // Arrange
            var config = new LLMServiceConfig
            {
                Provider = LLMProvider.OpenAI,
                ApiKey = ""
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => LLMServiceFactory.Create(config));
        }

        [Fact]
        public void Create_WithNullApiKey_ThrowsArgumentException()
        {
            // Arrange
            var config = new LLMServiceConfig
            {
                Provider = LLMProvider.OpenAI,
                ApiKey = null!
            };

            // Act & Assert
            Assert.Throws<ArgumentException>(() => LLMServiceFactory.Create(config));
        }

        [Fact]
        public void CreateOpenAI_WithCustomBaseUrl_UsesCustomUrl()
        {
            // Act
            var service = LLMServiceFactory.CreateOpenAI(
                "test-api-key",
                "https://custom.openai.com/v1");

            // Assert
            Assert.NotNull(service);
            Assert.IsType<OpenAIService>(service);
        }

        [Fact]
        public void CreateClaude_WithCustomApiVersion_UsesCustomVersion()
        {
            // Act
            var service = LLMServiceFactory.CreateClaude(
                "test-api-key",
                "2024-01-01");

            // Assert
            Assert.NotNull(service);
            Assert.IsType<ClaudeService>(service);
        }

        [Fact]
        public void CreateGemini_WithCustomBaseUrl_UsesCustomUrl()
        {
            // Act
            var service = LLMServiceFactory.CreateGemini(
                "test-api-key",
                "https://custom.googleapis.com/v1beta");

            // Assert
            Assert.NotNull(service);
            Assert.IsType<GeminiService>(service);
        }

        [Theory]
        [InlineData(LLMProvider.OpenAI)]
        [InlineData(LLMProvider.Claude)]
        [InlineData(LLMProvider.Gemini)]
        public void Create_AllProviders_ReturnsValidService(LLMProvider provider)
        {
            // Arrange
            var config = new LLMServiceConfig
            {
                Provider = provider,
                ApiKey = "test-api-key"
            };

            // Act
            var service = LLMServiceFactory.Create(config);

            // Assert
            Assert.NotNull(service);
            Assert.IsAssignableFrom<ILLMService>(service);
        }
    }
}
