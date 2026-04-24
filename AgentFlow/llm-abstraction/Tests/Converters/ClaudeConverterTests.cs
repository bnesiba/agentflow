using System.Collections.Generic;
using System.Linq;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Claude.Models;
using LLMAbstraction.Tests.Helpers;
using Xunit;

namespace LLMAbstraction.Tests.Converters
{
    public class ClaudeConverterTests
    {
        private readonly ClaudeConverter _converter;

        public ClaudeConverterTests()
        {
            _converter = new ClaudeConverter();
        }

        [Fact]
        public void ConvertRequest_SimpleTextMessage_ConvertsCorrectly()
        {
            // Arrange
            var request = TestHelpers.CreateSimpleRequest("claude-opus-4-6");

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("claude-opus-4-6", result.Model);
            Assert.Single(result.Messages);
            Assert.Equal("user", result.Messages[0].Role);
            Assert.Single(result.Messages[0].Content);
            Assert.Equal("text", result.Messages[0].Content[0].Type);
            Assert.Equal("Hello, world!", result.Messages[0].Content[0].Text);
            Assert.Equal(100, result.MaxTokens);
            Assert.Equal(0.7, result.Temperature);
        }

        [Fact]
        public void ConvertRequest_WithSystemMessage_ExtractsToSystemParameter()
        {
            // Arrange
            var request = TestHelpers.CreateRequestWithSystem("You are helpful.");

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal("You are helpful.", result.System);
            Assert.Single(result.Messages); // System not in messages
            Assert.Equal("user", result.Messages[0].Role);
        }

        [Fact]
        public void ConvertRequest_NoMaxTokens_SetsDefault1024()
        {
            // Arrange
            var request = TestHelpers.CreateSimpleRequest("claude-opus-4-6");
            request.Parameters.MaxTokens = null;

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.Equal(1024, result.MaxTokens);
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
        public void ConvertRequest_WithImageUrl_ConvertsToImageBlock()
        {
            // Arrange
            var request = new UnifiedRequest
            {
                Model = "claude-opus-4-6",
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
            Assert.Equal(2, result.Messages[0].Content.Count);
            Assert.Equal("text", result.Messages[0].Content[0].Type);
            Assert.Equal("image", result.Messages[0].Content[1].Type);
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
            Assert.Equal("get_weather", result.Tools[0].Name);
            Assert.NotNull(result.Tools[0].InputSchema);
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
            Assert.IsType<ClaudeToolChoice>(result.ToolChoice);
            Assert.Equal("auto", ((ClaudeToolChoice)result.ToolChoice!).Type);
        }

        [Fact]
        public void ConvertRequest_WithToolChoiceRequired_SetsToolChoiceAny()
        {
            // Arrange
            var request = TestHelpers.CreateToolCallRequest();
            request.ToolChoice = new ToolChoice { Type = ToolChoiceType.Required };

            // Act
            var result = _converter.ConvertRequest(request);

            // Assert
            Assert.IsType<ClaudeToolChoice>(result.ToolChoice);
            Assert.Equal("any", ((ClaudeToolChoice)result.ToolChoice!).Type);
        }

        [Fact]
        public void ConvertRequest_WithToolChoiceSpecific_SetsToolChoiceTool()
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
            Assert.IsType<ClaudeToolChoice>(result.ToolChoice);
            var toolChoice = (ClaudeToolChoice)result.ToolChoice!;
            Assert.Equal("tool", toolChoice.Type);
            Assert.Equal("get_weather", toolChoice.Name);
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
            Assert.Equal(40, result.TopK);
        }

        [Fact]
        public void ConvertResponse_SimpleTextResponse_ConvertsCorrectly()
        {
            // Arrange
            var claudeResponse = new ClaudeMessageResponse
            {
                Id = "msg_123",
                Model = "claude-opus-4-6",
                Role = "assistant",
                Content = new List<ClaudeContentBlock>
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

            // Act
            var result = _converter.ConvertResponse(claudeResponse);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("msg_123", result.Id);
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
        public void ConvertResponse_WithToolUse_ConvertsToolCallContent()
        {
            // Arrange
            var claudeResponse = new ClaudeMessageResponse
            {
                Id = "msg_123",
                Model = "claude-opus-4-6",
                Role = "assistant",
                Content = new List<ClaudeContentBlock>
                {
                    new ClaudeContentBlock
                    {
                        Type = "tool_use",
                        Id = "toolu_123",
                        Name = "get_weather",
                        Input = new Dictionary<string, object>
                        {
                            ["location"] = "Boston"
                        }
                    }
                },
                StopReason = "tool_use",
                Usage = new ClaudeUsage()
            };

            // Act
            var result = _converter.ConvertResponse(claudeResponse);

            // Assert
            Assert.Single(result.Choices[0].Message.Content);
            
            var toolCall = result.Choices[0].Message.Content[0] as ToolCallContent;
            Assert.NotNull(toolCall);
            Assert.Equal("toolu_123", toolCall.Id);
            Assert.Equal("get_weather", toolCall.Name);
            Assert.Contains("location", toolCall.Input.Keys);
            
            Assert.Equal(FinishReason.ToolCalls, result.Choices[0].FinishReason);
        }

        [Fact]
        public void ConvertResponse_StopReasonMaxTokens_ConvertsCorrectly()
        {
            // Arrange
            var claudeResponse = new ClaudeMessageResponse
            {
                Id = "msg_123",
                Model = "claude-opus-4-6",
                Role = "assistant",
                Content = new List<ClaudeContentBlock>
                {
                    new ClaudeContentBlock { Type = "text", Text = "Truncated..." }
                },
                StopReason = "max_tokens",
                Usage = new ClaudeUsage()
            };

            // Act
            var result = _converter.ConvertResponse(claudeResponse);

            // Assert
            Assert.Equal(FinishReason.MaxTokens, result.Choices[0].FinishReason);
        }

        [Fact]
        public void ConvertResponse_StopReasonStopSequence_ConvertsToStop()
        {
            // Arrange
            var claudeResponse = new ClaudeMessageResponse
            {
                Id = "msg_123",
                Model = "claude-opus-4-6",
                Role = "assistant",
                Content = new List<ClaudeContentBlock>
                {
                    new ClaudeContentBlock { Type = "text", Text = "Response" }
                },
                StopReason = "stop_sequence",
                Usage = new ClaudeUsage()
            };

            // Act
            var result = _converter.ConvertResponse(claudeResponse);

            // Assert
            Assert.Equal(FinishReason.Stop, result.Choices[0].FinishReason);
        }

        [Fact]
        public void ConvertResponse_WithCacheMetrics_IncludesInUsage()
        {
            // Arrange
            var claudeResponse = new ClaudeMessageResponse
            {
                Id = "msg_123",
                Model = "claude-opus-4-6",
                Role = "assistant",
                Content = new List<ClaudeContentBlock>
                {
                    new ClaudeContentBlock { Type = "text", Text = "Response" }
                },
                StopReason = "end_turn",
                Usage = new ClaudeUsage
                {
                    InputTokens = 100,
                    OutputTokens = 50,
                    CacheCreationInputTokens = 20,
                    CacheReadInputTokens = 30
                }
            };

            // Act
            var result = _converter.ConvertResponse(claudeResponse);

            // Assert
            Assert.Equal(100, result.Usage.PromptTokens);
            Assert.Equal(50, result.Usage.CompletionTokens);
            Assert.Equal(150, result.Usage.TotalTokens);
        }
    }
}
