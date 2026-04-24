using System;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Claude.Models;
using LLMAbstraction.Tests.Helpers;
using Moq;
using Moq.Protected;
using Xunit;

namespace LLMAbstraction.Tests.Services
{
    public class ClaudeServiceTests
    {
        private readonly JsonSerializerOptions _jsonOptions;

        public ClaudeServiceTests()
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
            var mockResponse = new ClaudeMessageResponse
            {
                Id = "msg_123",
                Model = "claude-opus-4-6",
                Role = "assistant",
                Content = new System.Collections.Generic.List<ClaudeContentBlock>
                {
                    new ClaudeContentBlock
                    {
                        Type = "text",
                        Text = "Hello! How can I help you?"
                    }
                },
                StopReason = "end_turn",
                Usage = new ClaudeUsage
                {
                    InputTokens = 10,
                    OutputTokens = 15
                }
            };

            var httpClient = CreateMockHttpClient(mockResponse);
            var service = new ClaudeService(httpClient, "test-api-key");
            var request = TestHelpers.CreateSimpleRequest("claude-opus-4-6");

            // Act
            var result = await service.GenerateAsync(request);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("msg_123", result.Id);
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
                BaseAddress = new Uri("https://api.anthropic.com")
            };
            var service = new ClaudeService(httpClient, "test-api-key");
            var request = TestHelpers.CreateSimpleRequest("claude-opus-4-6");

            // Act & Assert
            await Assert.ThrowsAsync<HttpRequestException>(
                async () => await service.GenerateAsync(request));
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new ClaudeService(null!));
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
                BaseAddress = new Uri("https://api.anthropic.com")
            };
        }
    }
}
