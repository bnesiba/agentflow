using System.Collections.Generic;
using System.Linq;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.Gemini.Models;
using LLMAbstraction.Tests.Helpers;
using Xunit;

namespace LLMAbstraction.Tests.Converters
{
    public class GeminiConverterTests
    {
        private readonly GeminiConverter _converter;

        public GeminiConverterTests()
        {
            _converter = new GeminiConverter();
        }

        [Fact]
        public void ConvertRequest_SimpleTextMessage_ConvertsCorrectly()
        {
            // Arrange
            var request = TestHelpers.CreateSimpleRequest("gemini-3-pro");

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result.Contents);
            Assert.Equal("user", result.Contents[0].Role);
            Assert.Single(result.Contents[0].Parts);
            Assert.Equal("Hello, world!", result.Contents[0].Parts[0].Text);
            Assert.NotNull(result.GenerationConfig);
            Assert.Equal(100, result.GenerationConfig!.MaxOutputTokens);
            Assert.Equal(0.7, result.GenerationConfig.Temperature);
        }

        [Fact]
        public void ConvertRequest_WithSystemMessage_ExtractsToSystemInstruction()
        {
            // Arrange
            var request = TestHelpers.CreateRequestWithSystem("You are helpful.");

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.SystemInstruction);
            Assert.Single(result.SystemInstruction!.Parts);
            Assert.Equal("You are helpful.", result.SystemInstruction.Parts[0].Text);
            Assert.Single(result.Contents); // System not in contents
            Assert.Equal("user", result.Contents[0].Role);
        }

        [Fact]
        public void ConvertRequest_MultiTurnConversation_ConvertsAllMessages()
        {
            // Arrange
            var request = TestHelpers.CreateMultiTurnRequest();

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal(3, result.Contents.Count);
            Assert.Equal("user", result.Contents[0].Role);
            Assert.Equal("model", result.Contents[1].Role); // Assistant -> model
            Assert.Equal("user", result.Contents[2].Role);
        }

        [Fact]
        public void ConvertRequest_WithImageBase64_ConvertsToPart()
        {
            // Arrange
            var request = new UnifiedRequest
            {
                Model = "gemini-3-pro",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, new List<ContentBlock>
                    {
                        new TextContent { Text = "What's in this image?" },
                        new ImageContent
                        {
                            Source = new ImageSource
                            {
                                Data = "base64data",
                                MediaType = "image/jpeg"
                            }
                        }
                    })
                },
                Parameters = new GenerationParameters()
            };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Single(result.Contents);
            Assert.Equal(2, result.Contents[0].Parts.Count);
            Assert.Equal("What's in this image?", result.Contents[0].Parts[0].Text);
            Assert.NotNull(result.Contents[0].Parts[1].InlineData);
            Assert.Equal("base64data", result.Contents[0].Parts[1].InlineData!.Data);
            Assert.Equal("image/jpeg", result.Contents[0].Parts[1].InlineData.MimeType);
        }

        [Fact]
        public void ConvertRequest_WithTools_ConvertsToolDefinitions()
        {
            // Arrange
            var request = TestHelpers.CreateToolCallRequest();

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.Tools);
            Assert.Single(result.Tools);
            Assert.Single(result.Tools![0].FunctionDeclarations);
            Assert.Equal("get_weather", result.Tools[0].FunctionDeclarations[0].Name);
        }

        [Fact]
        public void ConvertRequest_WithToolChoiceAuto_ConvertsToAUTO()
        {
            // Arrange
            var request = TestHelpers.CreateToolCallRequest();
            request.ToolChoice = new ToolChoice { Type = ToolChoiceType.Auto };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.ToolConfig);
            Assert.Equal("AUTO", result.ToolConfig!.FunctionCallingConfig.Mode);
        }

        [Fact]
        public void ConvertRequest_WithToolChoiceRequired_ConvertsToANY()
        {
            // Arrange
            var request = TestHelpers.CreateToolCallRequest();
            request.ToolChoice = new ToolChoice { Type = ToolChoiceType.Required };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.ToolConfig);
            Assert.Equal("ANY", result.ToolConfig!.FunctionCallingConfig.Mode);
        }

        [Fact]
        public void ConvertRequest_WithToolChoiceNone_ConvertsToNONE()
        {
            // Arrange
            var request = TestHelpers.CreateToolCallRequest();
            request.ToolChoice = new ToolChoice { Type = ToolChoiceType.None };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.ToolConfig);
            Assert.Equal("NONE", result.ToolConfig!.FunctionCallingConfig.Mode);
        }

        [Fact]
        public void ConvertRequest_WithTopK_IncludesTopK()
        {
            // Arrange
            var request = TestHelpers.CreateSimpleRequest();
            request.Parameters.TopK = 40;

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal(40, result.GenerationConfig!.TopK);
        }

        [Fact]
        public void ConvertRequest_WithStopSequences_IncludesStopSequences()
        {
            // Arrange
            var request = TestHelpers.CreateSimpleRequest();
            request.Parameters.StopSequences = new List<string> { "\n\n", "END" };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal(new[] { "\n\n", "END" }, result.GenerationConfig!.StopSequences);
        }

        [Fact]
        public void ConvertResponse_SimpleTextResponse_ConvertsCorrectly()
        {
            // Arrange
            var geminiResponse = new GeminiGenerateResponse
            {
                Candidates = new List<GeminiCandidate>
                {
                    new GeminiCandidate
                    {
                        Index = 0,
                        Content = new GeminiContent
                        {
                            Role = "model",
                            Parts = new List<GeminiPart>
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

            // Act
            var result = _converter.ConvertResponse(geminiResponse);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result.Choices);
            Assert.Equal(MessageRole.Assistant, result.Choices[0].Message.Role);
            Assert.Single(result.Choices[0].Message.Content);
            
            var textContent = result.Choices[0].Message.Content[0] as TextContent;
            Assert.NotNull(textContent);
            Assert.Equal("Hello! How can I help you?", textContent.Text);
            
            Assert.Equal(FinishReason.Stop, result.Choices[0].FinishReason);
            Assert.Equal(10, result.Usage.PromptTokens);
            Assert.Equal(15, result.Usage.CompletionTokens);
            Assert.Equal(25, result.Usage.TotalTokens);
        }

        [Fact]
        public void ConvertResponse_WithFunctionCall_ConvertsToolCallContent()
        {
            // Arrange
            var geminiResponse = new GeminiGenerateResponse
            {
                Candidates = new List<GeminiCandidate>
                {
                    new GeminiCandidate
                    {
                        Index = 0,
                        Content = new GeminiContent
                        {
                            Role = "model",
                            Parts = new List<GeminiPart>
                            {
                                new GeminiPart
                                {
                                    FunctionCall = new GeminiFunctionCall
                                    {
                                        Name = "get_weather",
                                        Args = new Dictionary<string, object>
                                        {
                                            ["location"] = "Boston"
                                        }
                                    }
                                }
                            }
                        },
                        FinishReason = "STOP"
                    }
                },
                UsageMetadata = new GeminiUsageMetadata
                {
                    PromptTokenCount = 20,
                    CandidatesTokenCount = 10,
                    TotalTokenCount = 30
                }
            };

            // Act
            var result = _converter.ConvertResponse(geminiResponse);

            // Assert
            Assert.Single(result.Choices[0].Message.Content);
            
            var toolCall = result.Choices[0].Message.Content[0] as ToolCallContent;
            Assert.NotNull(toolCall);
            Assert.Equal("get_weather", toolCall.Name);
            Assert.Contains("location", toolCall.Input.Keys);
            Assert.NotNull(toolCall.Id); // Generated ID
            Assert.NotEmpty(toolCall.Id);
        }

        [Fact]
        public void ConvertResponse_FinishReasonMAX_TOKENS_ConvertsToMaxTokens()
        {
            // Arrange
            var geminiResponse = new GeminiGenerateResponse
            {
                Candidates = new List<GeminiCandidate>
                {
                    new GeminiCandidate
                    {
                        Index = 0,
                        Content = new GeminiContent
                        {
                            Parts = new List<GeminiPart>
                            {
                                new GeminiPart { Text = "Truncated..." }
                            }
                        },
                        FinishReason = "MAX_TOKENS"
                    }
                },
                UsageMetadata = new GeminiUsageMetadata()
            };

            // Act
            var result = _converter.ConvertResponse(geminiResponse);

            // Assert
            Assert.Equal(FinishReason.MaxTokens, result.Choices[0].FinishReason);
        }

        [Fact]
        public void ConvertResponse_FinishReasonSAFETY_ConvertsToContentFilter()
        {
            // Arrange
            var geminiResponse = new GeminiGenerateResponse
            {
                Candidates = new List<GeminiCandidate>
                {
                    new GeminiCandidate
                    {
                        Index = 0,
                        Content = new GeminiContent
                        {
                            Parts = new List<GeminiPart>
                            {
                                new GeminiPart { Text = "Blocked" }
                            }
                        },
                        FinishReason = "SAFETY"
                    }
                },
                UsageMetadata = new GeminiUsageMetadata()
            };

            // Act
            var result = _converter.ConvertResponse(geminiResponse);

            // Assert
            Assert.Equal(FinishReason.ContentFilter, result.Choices[0].FinishReason);
        }

        [Fact]
        public void ConvertResponse_MultipleCandidates_ConvertsAll()
        {
            // Arrange
            var geminiResponse = new GeminiGenerateResponse
            {
                Candidates = new List<GeminiCandidate>
                {
                    new GeminiCandidate
                    {
                        Index = 0,
                        Content = new GeminiContent
                        {
                            Parts = new List<GeminiPart>
                            {
                                new GeminiPart { Text = "Response 1" }
                            }
                        },
                        FinishReason = "STOP"
                    },
                    new GeminiCandidate
                    {
                        Index = 1,
                        Content = new GeminiContent
                        {
                            Parts = new List<GeminiPart>
                            {
                                new GeminiPart { Text = "Response 2" }
                            }
                        },
                        FinishReason = "STOP"
                    }
                },
                UsageMetadata = new GeminiUsageMetadata()
            };

            // Act
            var result = _converter.ConvertResponse(geminiResponse);

            // Assert
            Assert.Equal(2, result.Choices.Count);
            Assert.Equal(0, result.Choices[0].Index);
            Assert.Equal(1, result.Choices[1].Index);
        }
    }
}
