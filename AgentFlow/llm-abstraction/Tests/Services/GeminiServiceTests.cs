using System;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.Gemini.Models;
using LLMAbstraction.Tests.Helpers;
using Moq;
using Moq.Protected;
using Xunit;

namespace LLMAbstraction.Tests.Services
{
    public class GeminiServiceTests
    {
        private readonly JsonSerializerOptions _jsonOptions;

        public GeminiServiceTests()
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
            var mockResponse = new GeminiGenerateResponse
            {
                Candidates = new System.Collections.Generic.List<GeminiCandidate>
                {
                    new GeminiCandidate
                    {
                        Index = 0,
                        Content = new GeminiContent
                        {
                            Role = "model",
                            Parts = new System.Collections.Generic.List<GeminiPart>
                            {
                                new GeminiPart { Text = "Hello! How can I help you?" }
                            }
                        },
                        FinishReason = "STOP"
                    }
                },
                UsageMetadata = new GeminiUsageMetadata
                {
                    PromptTokenCount = 10,
                    CandidatesTokenCount = 15,
                    TotalTokenCount = 25
                }
            };

            var httpClient = CreateMockHttpClient(mockResponse);
            var service = new GeminiService(httpClient, "test-api-key");
            var request = TestHelpers.CreateSimpleRequest("gemini-3-pro");

            // Act
            var result = await service.GenerateAsync(request);

            // Assert
            Assert.NotNull(result);
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
                BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta")
            };
            var service = new GeminiService(httpClient, "test-api-key");
            var request = TestHelpers.CreateSimpleRequest("gemini-3-pro");

            // Act & Assert
            await Assert.ThrowsAsync<HttpRequestException>(
                async () => await service.GenerateAsync(request));
        }

        [Fact]
        public void Constructor_WithNullApiKey_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new GeminiService(null!));
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
                BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta")
            };
        }
    }
}
