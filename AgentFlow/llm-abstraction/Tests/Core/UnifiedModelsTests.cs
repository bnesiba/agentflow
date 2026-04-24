using System.Collections.Generic;
using LLMAbstraction.Core.Models;
using Xunit;

namespace LLMAbstraction.Tests.Core
{
    public class UnifiedModelsTests
    {
        [Fact]
        public void UnifiedMessage_Constructor_WithRole_CreatesMessage()
        {
            // Act
            var message = new UnifiedMessage(MessageRole.User, "Hello");

            // Assert
            Assert.Equal(MessageRole.User, message.Role);
            Assert.Single(message.Content);
            Assert.IsType<TextContent>(message.Content[0]);
            Assert.Equal("Hello", ((TextContent)message.Content[0]).Text);
        }

        [Fact]
        public void UnifiedMessage_Constructor_WithContentBlocks_CreatesMessage()
        {
            // Arrange
            var blocks = new List<ContentBlock>
            {
                new TextContent { Text = "Hello" },
                new ImageContent { Source = new ImageSource { Url = "http://example.com/image.jpg" } }
            };

            // Act
            var message = new UnifiedMessage(MessageRole.User, blocks);

            // Assert
            Assert.Equal(MessageRole.User, message.Role);
            Assert.Equal(2, message.Content.Count);
            Assert.IsType<TextContent>(message.Content[0]);
            Assert.IsType<ImageContent>(message.Content[1]);
        }

        [Fact]
        public void TextContent_Type_ReturnsText()
        {
            // Arrange
            var content = new TextContent { Text = "Hello" };

            // Act & Assert
            Assert.Equal("text", content.Type);
        }

        [Fact]
        public void ImageContent_Type_ReturnsImage()
        {
            // Arrange
            var content = new ImageContent 
            { 
                Source = new ImageSource { Url = "http://example.com/image.jpg" } 
            };

            // Act & Assert
            Assert.Equal("image", content.Type);
        }

        [Fact]
        public void ToolCallContent_Type_ReturnsToolCall()
        {
            // Arrange
            var content = new ToolCallContent
            {
                Id = "call_123",
                Name = "get_weather",
                Input = new Dictionary<string, object>()
            };

            // Act & Assert
            Assert.Equal("tool_call", content.Type);
        }

        [Fact]
        public void ToolResultContent_Type_ReturnsToolResult()
        {
            // Arrange
            var content = new ToolResultContent
            {
                ToolCallId = "call_123",
                Output = "Result"
            };

            // Act & Assert
            Assert.Equal("tool_result", content.Type);
        }

        [Fact]
        public void GenerationParameters_DefaultValues_AreNull()
        {
            // Arrange & Act
            var parameters = new GenerationParameters();

            // Assert
            Assert.Null(parameters.MaxTokens);
            Assert.Null(parameters.Temperature);
            Assert.Null(parameters.TopP);
            Assert.Null(parameters.TopK);
            Assert.Null(parameters.StopSequences);
            Assert.False(parameters.Stream);
        }

        [Fact]
        public void ToolChoice_Auto_CreatesCorrectType()
        {
            // Act
            var toolChoice = new ToolChoice { Type = ToolChoiceType.Auto };

            // Assert
            Assert.Equal(ToolChoiceType.Auto, toolChoice.Type);
            Assert.Null(toolChoice.ToolName);
        }

        [Fact]
        public void ToolChoice_Specific_RequiresToolName()
        {
            // Act
            var toolChoice = new ToolChoice 
            { 
                Type = ToolChoiceType.Specific,
                ToolName = "get_weather"
            };

            // Assert
            Assert.Equal(ToolChoiceType.Specific, toolChoice.Type);
            Assert.Equal("get_weather", toolChoice.ToolName);
        }

        [Fact]
        public void ToolDefinition_HasRequiredFields()
        {
            // Arrange & Act
            var tool = new ToolDefinition
            {
                Name = "get_weather",
                Description = "Get weather information",
                Parameters = new Dictionary<string, object>
                {
                    ["type"] = "object"
                }
            };

            // Assert
            Assert.Equal("get_weather", tool.Name);
            Assert.Equal("Get weather information", tool.Description);
            Assert.True(tool.Parameters.ContainsKey("type"));
        }

        [Fact]
        public void UnifiedRequest_DefaultValues_AreSet()
        {
            // Arrange & Act
            var request = new UnifiedRequest();

            // Assert
            Assert.Empty(request.Model);
            Assert.NotNull(request.Messages);
            Assert.Empty(request.Messages);
            Assert.NotNull(request.Parameters);
        }

        [Fact]
        public void UnifiedResponse_DefaultValues_AreSet()
        {
            // Arrange & Act
            var response = new UnifiedResponse();

            // Assert
            Assert.Empty(response.Id);
            Assert.Empty(response.Model);
            Assert.NotNull(response.Choices);
            Assert.Empty(response.Choices);
            Assert.NotNull(response.Usage);
        }

        [Fact]
        public void UsageInfo_CalculatesTotalTokens()
        {
            // Arrange & Act
            var usage = new UsageInfo
            {
                PromptTokens = 10,
                CompletionTokens = 20,
                TotalTokens = 30
            };

            // Assert
            Assert.Equal(10, usage.PromptTokens);
            Assert.Equal(20, usage.CompletionTokens);
            Assert.Equal(30, usage.TotalTokens);
        }

        [Fact]
        public void ImageSource_SupportsUrl()
        {
            // Arrange & Act
            var source = new ImageSource
            {
                Url = "https://example.com/image.jpg"
            };

            // Assert
            Assert.Equal("https://example.com/image.jpg", source.Url);
            Assert.Null(source.Data);
            Assert.Null(source.MediaType);
        }

        [Fact]
        public void ImageSource_SupportsBase64()
        {
            // Arrange & Act
            var source = new ImageSource
            {
                Data = "base64data",
                MediaType = "image/jpeg"
            };

            // Assert
            Assert.Equal("base64data", source.Data);
            Assert.Equal("image/jpeg", source.MediaType);
            Assert.Null(source.Url);
        }

        [Fact]
        public void FinishReason_AllValues_AreDefined()
        {
            // Assert
            Assert.Equal(FinishReason.Stop, FinishReason.Stop);
            Assert.Equal(FinishReason.MaxTokens, FinishReason.MaxTokens);
            Assert.Equal(FinishReason.ToolCalls, FinishReason.ToolCalls);
            Assert.Equal(FinishReason.ContentFilter, FinishReason.ContentFilter);
            Assert.Equal(FinishReason.Other, FinishReason.Other);
        }

        [Fact]
        public void MessageRole_AllValues_AreDefined()
        {
            // Assert
            Assert.Equal(MessageRole.System, MessageRole.System);
            Assert.Equal(MessageRole.User, MessageRole.User);
            Assert.Equal(MessageRole.Assistant, MessageRole.Assistant);
            Assert.Equal(MessageRole.Tool, MessageRole.Tool);
        }
    }
}
