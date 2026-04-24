using System.Collections.Generic;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Tests.Helpers
{
    /// <summary>
    /// Helper methods for creating test data
    /// </summary>
    public static class TestHelpers
    {
        public static UnifiedRequest CreateSimpleRequest(string model = "test-model")
        {
            return new UnifiedRequest
            {
                Model = model,
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Hello, world!")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 100,
                    Temperature = 0.7
                }
            };
        }

        public static UnifiedRequest CreateRequestWithSystem(string systemMessage = "You are helpful.")
        {
            return new UnifiedRequest
            {
                Model = "test-model",
                System = systemMessage,
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Hello!")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 100
                }
            };
        }

        public static UnifiedRequest CreateMultiTurnRequest()
        {
            return new UnifiedRequest
            {
                Model = "test-model",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "What is 2+2?"),
                    new UnifiedMessage(MessageRole.Assistant, "2+2 equals 4."),
                    new UnifiedMessage(MessageRole.User, "What about 3+3?")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 100,
                    Temperature = 0.5
                }
            };
        }

        public static UnifiedRequest CreateMultimodalRequest()
        {
            return new UnifiedRequest
            {
                Model = "test-model",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, new List<ContentBlock>
                    {
                        new TextContent { Text = "What's in this image?" },
                        new ImageContent
                        {
                            Source = new ImageSource
                            {
                                Url = "https://example.com/image.jpg"
                            }
                        }
                    })
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 200
                }
            };
        }

        public static UnifiedRequest CreateToolCallRequest()
        {
            var tool = new ToolDefinition
            {
                Name = "get_weather",
                Description = "Get the current weather",
                Parameters = new Dictionary<string, object>
                {
                    ["type"] = "object",
                    ["properties"] = new Dictionary<string, object>
                    {
                        ["location"] = new Dictionary<string, object>
                        {
                            ["type"] = "string",
                            ["description"] = "The city name"
                        }
                    },
                    ["required"] = new[] { "location" }
                }
            };

            return new UnifiedRequest
            {
                Model = "test-model",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "What's the weather in Boston?")
                },
                Tools = new List<ToolDefinition> { tool },
                ToolChoice = new ToolChoice { Type = ToolChoiceType.Auto },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 100
                }
            };
        }

        public static UnifiedResponse CreateSimpleResponse()
        {
            return new UnifiedResponse
            {
                Id = "test-response-id",
                Model = "test-model",
                Choices = new List<ResponseChoice>
                {
                    new ResponseChoice
                    {
                        Index = 0,
                        Message = new UnifiedMessage(MessageRole.Assistant, "Hello! How can I help you?"),
                        FinishReason = FinishReason.Stop
                    }
                },
                Usage = new UsageInfo
                {
                    PromptTokens = 10,
                    CompletionTokens = 15,
                    TotalTokens = 25
                }
            };
        }

        public static UnifiedResponse CreateToolCallResponse()
        {
            return new UnifiedResponse
            {
                Id = "test-response-id",
                Model = "test-model",
                Choices = new List<ResponseChoice>
                {
                    new ResponseChoice
                    {
                        Index = 0,
                        Message = new UnifiedMessage(MessageRole.Assistant, new List<ContentBlock>
                        {
                            new ToolCallContent
                            {
                                Id = "call_123",
                                Name = "get_weather",
                                Input = new Dictionary<string, object>
                                {
                                    ["location"] = "Boston"
                                }
                            }
                        }),
                        FinishReason = FinishReason.ToolCalls
                    }
                },
                Usage = new UsageInfo
                {
                    PromptTokens = 20,
                    CompletionTokens = 10,
                    TotalTokens = 30
                }
            };
        }
    }
}
