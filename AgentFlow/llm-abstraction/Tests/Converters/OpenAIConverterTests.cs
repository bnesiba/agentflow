using System.Collections.Generic;
using System.Linq;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.OpenAI.Models;
using LLMAbstraction.Tests.Helpers;
using Xunit;

namespace LLMAbstraction.Tests.Converters
{
    public class OpenAIConverterTests
    {
        private readonly OpenAIConverter _converter;

        public OpenAIConverterTests()
        {
            _converter = new OpenAIConverter();
        }

        [Fact]
        public void ConvertRequest_SimpleTextMessage_ConvertsCorrectly()
        {
            // Arrange
            var request = TestHelpers.CreateSimpleRequest("gpt-4o");

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("gpt-4o", result.Model);
            Assert.Single(result.Messages);
            Assert.Equal("user", result.Messages[0].Role);
            Assert.Equal("Hello, world!", result.Messages[0].Content);
            Assert.Equal(100, result.MaxTokens);
            Assert.Equal(0.7, result.Temperature);
        }

        [Fact]
        public void ConvertRequest_WithSystemMessage_AddsSystemMessage()
        {
            // Arrange
            var request = TestHelpers.CreateRequestWithSystem("You are helpful.");

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal(2, result.Messages.Count);
            Assert.Equal("system", result.Messages[0].Role);
            Assert.Equal("You are helpful.", result.Messages[0].Content);
            Assert.Equal("user", result.Messages[1].Role);
        }

        [Fact]
        public void ConvertRequest_MultiTurnConversation_ConvertsAllMessages()
        {
            // Arrange
            var request = TestHelpers.CreateMultiTurnRequest();

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal(3, result.Messages.Count);
            Assert.Equal("user", result.Messages[0].Role);
            Assert.Equal("assistant", result.Messages[1].Role);
            Assert.Equal("user", result.Messages[2].Role);
        }

        [Fact]
        public void ConvertRequest_WithImageUrl_ConvertsToContentArray()
        {
            // Arrange
            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, new List<ContentBlock>
                    {
                        new TextContent { Text = "What's in this image?" },
                        new ImageContent
                        {
                            Source = new ImageSource { Url = "https://example.com/image.jpg" }
                        }
                    })
                },
                Parameters = new GenerationParameters()
            };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Single(result.Messages);
            Assert.IsType<List<object>>(result.Messages[0].Content);
            var contentArray = (List<object>)result.Messages[0].Content!;
            Assert.Equal(2, contentArray.Count);
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
            Assert.Equal("function", result.Tools[0].Type);
            Assert.Equal("get_weather", result.Tools[0].Function.Name);
        }

        [Fact]
        public void ConvertRequest_WithToolChoiceAuto_SetsToolChoiceAuto()
        {
            // Arrange
            var request = TestHelpers.CreateToolCallRequest();
            request.ToolChoice = new ToolChoice { Type = ToolChoiceType.Auto };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal("auto", result.ToolChoice);
        }

        [Fact]
        public void ConvertRequest_WithToolChoiceRequired_SetsToolChoiceRequired()
        {
            // Arrange
            var request = TestHelpers.CreateToolCallRequest();
            request.ToolChoice = new ToolChoice { Type = ToolChoiceType.Required };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal("required", result.ToolChoice);
        }

        [Fact]
        public void ConvertRequest_WithToolChoiceSpecific_SetsToolChoiceObject()
        {
            // Arrange
            var request = TestHelpers.CreateToolCallRequest();
            request.ToolChoice = new ToolChoice 
            { 
                Type = ToolChoiceType.Specific,
                ToolName = "get_weather"
            };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.IsType<object>(result.ToolChoice);
        }

        [Fact]
        public void ConvertResponse_SimpleTextResponse_ConvertsCorrectly()
        {
            // Arrange
            var openAIResponse = new OpenAIChatResponse
            {
                Id = "chatcmpl-123",
                Model = "gpt-4o",
                Choices = new List<OpenAIChoice>
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

            // Act
            var result = _converter.ConvertResponse(openAIResponse);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("chatcmpl-123", result.Id);
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
        public void ConvertResponse_WithToolCalls_ConvertsToolCallContent()
        {
            // Arrange
            var openAIResponse = new OpenAIChatResponse
            {
                Id = "chatcmpl-123",
                Model = "gpt-4o",
                Choices = new List<OpenAIChoice>
                {
                    new OpenAIChoice
                    {
                        Index = 0,
                        Message = new OpenAIMessage
                        {
                            Role = "assistant",
                            ToolCalls = new List<OpenAIToolCall>
                            {
                                new OpenAIToolCall
                                {
                                    Id = "call_123",
                                    Type = "function",
                                    Function = new OpenAIFunctionCall
                                    {
                                        Name = "get_weather",
                                        Arguments = "{\"location\":\"Boston\"}"
                                    }
                                }
                            }
                        },
                        FinishReason = "tool_calls"
                    }
                },
                Usage = new OpenAIUsage()
            };

            // Act
            var result = _converter.ConvertResponse(openAIResponse);

            // Assert
            Assert.Single(result.Choices[0].Message.Content);
            
            var toolCall = result.Choices[0].Message.Content[0] as ToolCallContent;
            Assert.NotNull(toolCall);
            Assert.Equal("call_123", toolCall.Id);
            Assert.Equal("get_weather", toolCall.Name);
            Assert.Contains("location", toolCall.Input.Keys);
            
            Assert.Equal(FinishReason.ToolCalls, result.Choices[0].FinishReason);
        }

        [Fact]
        public void ConvertResponse_FinishReasonLength_ConvertsToMaxTokens()
        {
            // Arrange
            var openAIResponse = new OpenAIChatResponse
            {
                Id = "chatcmpl-123",
                Model = "gpt-4o",
                Choices = new List<OpenAIChoice>
                {
                    new OpenAIChoice
                    {
                        Index = 0,
                        Message = new OpenAIMessage
                        {
                            Role = "assistant",
                            Content = "Truncated..."
                        },
                        FinishReason = "length"
                    }
                },
                Usage = new OpenAIUsage()
            };

            // Act
            var result = _converter.ConvertResponse(openAIResponse);

            // Assert
            Assert.Equal(FinishReason.MaxTokens, result.Choices[0].FinishReason);
        }

        [Fact]
        public void ConvertResponse_FinishReasonContentFilter_ConvertsCorrectly()
        {
            // Arrange
            var openAIResponse = new OpenAIChatResponse
            {
                Id = "chatcmpl-123",
                Model = "gpt-4o",
                Choices = new List<OpenAIChoice>
                {
                    new OpenAIChoice
                    {
                        Index = 0,
                        Message = new OpenAIMessage
                        {
                            Role = "assistant",
                            Content = "Filtered"
                        },
                        FinishReason = "content_filter"
                    }
                },
                Usage = new OpenAIUsage()
            };

            // Act
            var result = _converter.ConvertResponse(openAIResponse);

            // Assert
            Assert.Equal(FinishReason.ContentFilter, result.Choices[0].FinishReason);
        }

        [Fact]
        public void ConvertResponse_MultipleChoices_ConvertsAll()
        {
            // Arrange
            var openAIResponse = new OpenAIChatResponse
            {
                Id = "chatcmpl-123",
                Model = "gpt-4o",
                Choices = new List<OpenAIChoice>
                {
                    new OpenAIChoice
                    {
                        Index = 0,
                        Message = new OpenAIMessage { Role = "assistant", Content = "Response 1" },
                        FinishReason = "stop"
                    },
                    new OpenAIChoice
                    {
                        Index = 1,
                        Message = new OpenAIMessage { Role = "assistant", Content = "Response 2" },
                        FinishReason = "stop"
                    }
                },
                Usage = new OpenAIUsage()
            };

            // Act
            var result = _converter.ConvertResponse(openAIResponse);

            // Assert
            Assert.Equal(2, result.Choices.Count);
            Assert.Equal(0, result.Choices[0].Index);
            Assert.Equal(1, result.Choices[1].Index);
        }
    }
}
