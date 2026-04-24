using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.OpenAI.Models;
using Xunit;

namespace LLMAbstraction.Tests.Features
{
    public class StreamingTests
    {
        [Fact]
        public void StreamChunk_Construction_SetsPropertiesCorrectly()
        {
            // Arrange & Act
            var chunk = new StreamChunk
            {
                Id = "chunk-123",
                Model = "gpt-4o",
                ChoiceIndex = 0,
                Delta = new StreamDelta
                {
                    Role = MessageRole.Assistant,
                    Content = "Hello"
                },
                FinishReason = FinishReason.Stop,
                Usage = new UsageInfo
                {
                    PromptTokens = 10,
                    CompletionTokens = 5,
                    TotalTokens = 15
                }
            };

            // Assert
            Assert.Equal("chunk-123", chunk.Id);
            Assert.Equal("gpt-4o", chunk.Model);
            Assert.Equal(0, chunk.ChoiceIndex);
            Assert.Equal(MessageRole.Assistant, chunk.Delta.Role);
            Assert.Equal("Hello", chunk.Delta.Content);
            Assert.Equal(FinishReason.Stop, chunk.FinishReason);
            Assert.NotNull(chunk.Usage);
            Assert.Equal(10, chunk.Usage.PromptTokens);
        }

        [Fact]
        public void StreamDelta_WithToolCalls_ConstructsCorrectly()
        {
            // Arrange & Act
            var delta = new StreamDelta
            {
                ToolCalls = new List<ToolCallDelta>
                {
                    new ToolCallDelta
                    {
                        Index = 0,
                        Id = "call_123",
                        Name = "get_weather",
                        Arguments = "{\"location\":"
                    }
                }
            };

            // Assert
            Assert.Single(delta.ToolCalls);
            Assert.Equal(0, delta.ToolCalls[0].Index);
            Assert.Equal("call_123", delta.ToolCalls[0].Id);
            Assert.Equal("get_weather", delta.ToolCalls[0].Name);
            Assert.Contains("location", delta.ToolCalls[0].Arguments);
        }

        [Fact]
        public void ToolCallDelta_PartialArguments_HandlesCorrectly()
        {
            // Arrange & Act
            var delta1 = new ToolCallDelta
            {
                Index = 0,
                Id = "call_123",
                Name = "get_weather",
                Arguments = "{\"loc"
            };

            var delta2 = new ToolCallDelta
            {
                Index = 0,
                Arguments = "ation\":"
            };

            var delta3 = new ToolCallDelta
            {
                Index = 0,
                Arguments = "\"Boston\"}"
            };

            // Assert
            Assert.Equal("call_123", delta1.Id);
            Assert.Equal("get_weather", delta1.Name);
            Assert.Null(delta2.Id); // Only in first chunk
            Assert.Null(delta2.Name); // Only in first chunk
            
            // In real usage, these would be concatenated
            var fullArgs = delta1.Arguments + delta2.Arguments + delta3.Arguments;
            Assert.Equal("{\"location\":\"Boston\"}", fullArgs);
        }

        [Fact]
        public void OpenAIStreamChunk_Deserialization_WorksCorrectly()
        {
            // Arrange
            var json = @"{
                ""id"": ""chatcmpl-123"",
                ""object"": ""chat.completion.chunk"",
                ""created"": 1234567890,
                ""model"": ""gpt-4o"",
                ""choices"": [
                    {
                        ""index"": 0,
                        ""delta"": {
                            ""role"": ""assistant"",
                            ""content"": ""Hello""
                        },
                        ""finish_reason"": null
                    }
                ]
            }";

            // Act
            var chunk = System.Text.Json.JsonSerializer.Deserialize<OpenAIStreamChunk>(
                json,
                new System.Text.Json.JsonSerializerOptions
                {
                    PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
                });

            // Assert
            Assert.NotNull(chunk);
            Assert.Equal("chatcmpl-123", chunk.Id);
            Assert.Equal("gpt-4o", chunk.Model);
            Assert.Single(chunk.Choices);
            Assert.Equal("Hello", chunk.Choices[0].Delta.Content);
        }

        [Fact]
        public void StreamChunk_MultipleChoices_HandlesCorrectly()
        {
            // Arrange & Act
            var chunks = new List<StreamChunk>
            {
                new StreamChunk
                {
                    Id = "chunk-123",
                    Model = "gpt-4o",
                    ChoiceIndex = 0,
                    Delta = new StreamDelta { Content = "Response 1" }
                },
                new StreamChunk
                {
                    Id = "chunk-123",
                    Model = "gpt-4o",
                    ChoiceIndex = 1,
                    Delta = new StreamDelta { Content = "Response 2" }
                }
            };

            // Assert
            Assert.Equal(2, chunks.Count);
            Assert.Equal(0, chunks[0].ChoiceIndex);
            Assert.Equal(1, chunks[1].ChoiceIndex);
            Assert.Equal("Response 1", chunks[0].Delta.Content);
            Assert.Equal("Response 2", chunks[1].Delta.Content);
        }

        [Fact]
        public void StreamChunk_FinalChunk_ContainsFinishReason()
        {
            // Arrange & Act
            var finalChunk = new StreamChunk
            {
                Id = "chunk-123",
                Model = "gpt-4o",
                ChoiceIndex = 0,
                Delta = new StreamDelta(), // Empty delta
                FinishReason = FinishReason.Stop,
                Usage = new UsageInfo
                {
                    PromptTokens = 10,
                    CompletionTokens = 20,
                    TotalTokens = 30
                }
            };

            // Assert
            Assert.Equal(FinishReason.Stop, finalChunk.FinishReason);
            Assert.NotNull(finalChunk.Usage);
            Assert.Null(finalChunk.Delta.Content); // No content in final chunk
        }

        [Fact]
        public async Task StreamChunks_Aggregation_ReconstructsFullResponse()
        {
            // Arrange
            var chunks = new List<StreamChunk>
            {
                new StreamChunk
                {
                    Id = "chunk-123",
                    Model = "gpt-4o",
                    ChoiceIndex = 0,
                    Delta = new StreamDelta { Role = MessageRole.Assistant }
                },
                new StreamChunk
                {
                    Id = "chunk-123",
                    Model = "gpt-4o",
                    ChoiceIndex = 0,
                    Delta = new StreamDelta { Content = "Hello" }
                },
                new StreamChunk
                {
                    Id = "chunk-123",
                    Model = "gpt-4o",
                    ChoiceIndex = 0,
                    Delta = new StreamDelta { Content = " world" }
                },
                new StreamChunk
                {
                    Id = "chunk-123",
                    Model = "gpt-4o",
                    ChoiceIndex = 0,
                    Delta = new StreamDelta { Content = "!" }
                },
                new StreamChunk
                {
                    Id = "chunk-123",
                    Model = "gpt-4o",
                    ChoiceIndex = 0,
                    Delta = new StreamDelta(),
                    FinishReason = FinishReason.Stop
                }
            };

            // Act
            var contentChunks = chunks
                .Where(c => !string.IsNullOrEmpty(c.Delta.Content))
                .Select(c => c.Delta.Content)
                .ToList();
            var fullContent = string.Join("", contentChunks);

            // Assert
            Assert.Equal("Hello world!", fullContent);
            Assert.Equal(FinishReason.Stop, chunks.Last().FinishReason);
        }
    }
}
