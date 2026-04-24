using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;
using Xunit;

namespace LLMAbstraction.Tests.Integration
{
    /// <summary>
    /// End-to-end integration tests that require real API keys.
    /// These tests are skipped by default and should be run manually with valid API keys.
    /// Set environment variables: OPENAI_API_KEY, CLAUDE_API_KEY, GEMINI_API_KEY
    /// </summary>
    public class EndToEndTests
    {
        private readonly string? _openAIKey;
        private readonly string? _claudeKey;
        private readonly string? _geminiKey;

        public EndToEndTests()
        {
            _openAIKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            _claudeKey = Environment.GetEnvironmentVariable("CLAUDE_API_KEY");
            _geminiKey = Environment.GetEnvironmentVariable("GEMINI_API_KEY");
        }

        [Fact(Skip = "Requires real API key")]
        public async Task OpenAI_SimpleGeneration_ReturnsValidResponse()
        {
            // Arrange
            if (string.IsNullOrEmpty(_openAIKey))
                return;

            var service = LLMServiceFactory.CreateOpenAI(_openAIKey);
            var request = new UnifiedRequest
            {
                Model = "gpt-4o-mini",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Say 'Hello World' and nothing else.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 10,
                    Temperature = 0.0
                }
            };

            // Act
            var response = await service.GenerateAsync(request);

            // Assert
            Assert.NotNull(response);
            Assert.Single(response.Choices);
            Assert.NotEmpty(response.Choices[0].Message.Content);
            Assert.True(response.Usage.TotalTokens > 0);
            
            var textContent = response.Choices[0].Message.Content[0] as TextContent;
            Assert.NotNull(textContent);
            Assert.Contains("Hello", textContent.Text);
        }

        [Fact(Skip = "Requires real API key")]
        public async Task Claude_SimpleGeneration_ReturnsValidResponse()
        {
            // Arrange
            if (string.IsNullOrEmpty(_claudeKey))
                return;

            var service = LLMServiceFactory.CreateClaude(_claudeKey);
            var request = new UnifiedRequest
            {
                Model = "claude-opus-4-6",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Say 'Hello World' and nothing else.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 10,
                    Temperature = 0.0
                }
            };

            // Act
            var response = await service.GenerateAsync(request);

            // Assert
            Assert.NotNull(response);
            Assert.Single(response.Choices);
            Assert.NotEmpty(response.Choices[0].Message.Content);
            Assert.True(response.Usage.TotalTokens > 0);
            
            var textContent = response.Choices[0].Message.Content[0] as TextContent;
            Assert.NotNull(textContent);
            Assert.Contains("Hello", textContent.Text);
        }

        [Fact(Skip = "Requires real API key")]
        public async Task Gemini_SimpleGeneration_ReturnsValidResponse()
        {
            // Arrange
            if (string.IsNullOrEmpty(_geminiKey))
                return;

            var service = LLMServiceFactory.CreateGemini(_geminiKey);
            var request = new UnifiedRequest
            {
                Model = "gemini-2.5-flash",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Say 'Hello World' and nothing else.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 10,
                    Temperature = 0.0
                }
            };

            // Act
            var response = await service.GenerateAsync(request);

            // Assert
            Assert.NotNull(response);
            Assert.Single(response.Choices);
            Assert.NotEmpty(response.Choices[0].Message.Content);
            Assert.True(response.Usage.TotalTokens > 0);
            
            var textContent = response.Choices[0].Message.Content[0] as TextContent;
            Assert.NotNull(textContent);
            Assert.Contains("Hello", textContent.Text);
        }

        [Fact(Skip = "Requires real API key")]
        public async Task AllProviders_SameRequest_ProduceSimilarResults()
        {
            // Arrange
            if (string.IsNullOrEmpty(_openAIKey) || 
                string.IsNullOrEmpty(_claudeKey) || 
                string.IsNullOrEmpty(_geminiKey))
                return;

            var prompt = "What is 2+2? Answer with just the number.";
            var baseRequest = new UnifiedRequest
            {
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, prompt)
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 5,
                    Temperature = 0.0
                }
            };

            // Act
            var openAIService = LLMServiceFactory.CreateOpenAI(_openAIKey);
            var openAIRequest = baseRequest with { Model = "gpt-4o-mini" };
            var openAIResponse = await openAIService.GenerateAsync(openAIRequest);

            var claudeService = LLMServiceFactory.CreateClaude(_claudeKey);
            var claudeRequest = baseRequest with { Model = "claude-opus-4-6" };
            var claudeResponse = await claudeService.GenerateAsync(claudeRequest);

            var geminiService = LLMServiceFactory.CreateGemini(_geminiKey);
            var geminiRequest = baseRequest with { Model = "gemini-2.5-flash" };
            var geminiResponse = await geminiService.GenerateAsync(geminiRequest);

            // Assert
            var openAIText = (openAIResponse.Choices[0].Message.Content[0] as TextContent)?.Text;
            var claudeText = (claudeResponse.Choices[0].Message.Content[0] as TextContent)?.Text;
            var geminiText = (geminiResponse.Choices[0].Message.Content[0] as TextContent)?.Text;

            Assert.Contains("4", openAIText);
            Assert.Contains("4", claudeText);
            Assert.Contains("4", geminiText);
        }

        [Fact(Skip = "Requires real API key")]
        public async Task MultiTurnConversation_MaintainsContext()
        {
            // Arrange
            if (string.IsNullOrEmpty(_openAIKey))
                return;

            var service = LLMServiceFactory.CreateOpenAI(_openAIKey);
            var request = new UnifiedRequest
            {
                Model = "gpt-4o-mini",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "My name is Alice."),
                    new UnifiedMessage(MessageRole.Assistant, "Hello Alice! Nice to meet you."),
                    new UnifiedMessage(MessageRole.User, "What is my name?")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 20,
                    Temperature = 0.0
                }
            };

            // Act
            var response = await service.GenerateAsync(request);

            // Assert
            var textContent = response.Choices[0].Message.Content[0] as TextContent;
            Assert.NotNull(textContent);
            Assert.Contains("Alice", textContent.Text);
        }
    }
}
