using System;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.OpenAI.Models;
using LLMAbstraction.Tests.Helpers;
using Moq;
using Moq.Protected;
using Xunit;

namespace LLMAbstraction.Tests.Services
{
    public class OpenAIServiceTests
    {
        private readonly JsonSerializerOptions _jsonOptions;

        public OpenAIServiceTests()
        {
            _jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
        }

        [Fact]
        public async Task GenerateAsync_SimpleRequest_ReturnsResponse()
        {
            // Arrange
            var mockResponse = new OpenAIChatResponse
            {
                Id = "chatcmpl-123",
                Model = "gpt-4o",
                Choices = new System.Collections.Generic.List<OpenAIChoice>
                {
                    new OpenAIChoice
                    {
                        Index = 0,
                        Message = new OpenAIMessage
                        {
                            Role = "assistant",
                            Content = "Hello! How can I help you?"
                        },
                        FinishReason = "stop"
                    }
                },
                Usage = new OpenAIUsage
                {
                    PromptTokens = 10,
                    CompletionTokens = 15,
                    TotalTokens = 25
                }
            };

            var httpClient = CreateMockHttpClient(mockResponse);
            var service = new OpenAIService(httpClient, "test-api-key");
            var request = TestHelpers.CreateSimpleRequest("gpt-4o");

            // Act
            var result = await service.GenerateAsync(request);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("chatcmpl-123", result.Id);
            Assert.Single(result.Choices);
            var textContent = result.Choices[0].Message.Content[0] as TextContent;
            Assert.Equal("Hello! How can I help you?", textContent!.Text);
        }

        [Fact]
        public async Task GenerateAsync_ApiError_ThrowsHttpRequestException()
        {
            // Arrange
            var mockHandler = new Mock<HttpMessageHandler>();
            mockHandler.Protected()
                .Setup<Task<HttpResponseMessage>>(
                    "SendAsync",
                    ItExpr.IsAny<HttpRequestMessage>(),
                    ItExpr.IsAny<CancellationToken>())
                .ReturnsAsync(new HttpResponseMessage
                {
                    StatusCode = HttpStatusCode.BadRequest,
                    Content = new StringContent("{\"error\":{\"message\":\"Invalid request\"}}")
                });

            var httpClient = new HttpClient(mockHandler.Object)
            {
                BaseAddress = new Uri("https://api.openai.com/v1")
            };
            var service = new OpenAIService(httpClient, "test-api-key");
            var request = TestHelpers.CreateSimpleRequest("gpt-4o");

            // Act & Assert
            await Assert.ThrowsAsync<HttpRequestException>(
                async () => await service.GenerateAsync(request));
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new OpenAIService(null!));
        }

        private HttpClient CreateMockHttpClient(object responseObject)
        {
            var responseJson = JsonSerializer.Serialize(responseObject, _jsonOptions);
            var mockHandler = new Mock<HttpMessageHandler>();
            
            mockHandler.Protected()
                .Setup<Task<HttpResponseMessage>>(
                    "SendAsync",
                    ItExpr.IsAny<HttpRequestMessage>(),
                    ItExpr.IsAny<CancellationToken>())
                .ReturnsAsync(new HttpResponseMessage
                {
                    StatusCode = HttpStatusCode.OK,
                    Content = new StringContent(responseJson)
                });

            return new HttpClient(mockHandler.Object)
            {
                BaseAddress = new Uri("https://api.openai.com/v1")
            };
        }
    }
}
