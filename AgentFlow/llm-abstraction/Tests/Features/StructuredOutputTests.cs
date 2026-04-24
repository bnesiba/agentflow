using System.Collections.Generic;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using Xunit;

namespace LLMAbstraction.Tests.Features
{
    public class StructuredOutputTests
    {
        [Fact]
        public void OpenAI_StructuredOutput_JsonMode_ConvertsCorrectly()
        {
            // Arrange
            var converter = new OpenAIConverter();
            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Return JSON")
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.Json
                }
            };

            // Act
            var result = converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.ResponseFormat);
        }

        [Fact]
        public void OpenAI_StructuredOutput_JsonSchema_ConvertsCorrectly()
        {
            // Arrange
            var converter = new OpenAIConverter();
            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Return structured data")
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.JsonSchema,
                    JsonSchema = new JsonSchema
                    {
                        Name = "person",
                        Description = "A person object",
                        Schema = new Dictionary<string, object>
                        {
                            ["type"] = "object",
                            ["properties"] = new Dictionary<string, object>
                            {
                                ["name"] = new Dictionary<string, object> { ["type"] = "string" },
                                ["age"] = new Dictionary<string, object> { ["type"] = "number" }
                            },
                            ["required"] = new[] { "name", "age" }
                        },
                        Strict = true
                    }
                }
            };

            // Act
            var result = converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.ResponseFormat);
        }

        [Fact]
        public void Claude_StructuredOutput_SetsOutputConfig()
        {
            // Arrange
            var converter = new ClaudeConverter();
            var request = new UnifiedRequest
            {
                Model = "claude-opus-4-6",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Return JSON")
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.Json
                }
            };

            // Act
            var result = converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.OutputConfig);
            Assert.NotNull(result.OutputConfig.Format);
            Assert.Equal("json", result.OutputConfig.Format.Type);
        }

        [Fact]
        public void Claude_StructuredOutput_JsonSchema_SetsOutputConfigWithSchema()
        {
            // Arrange
            var converter = new ClaudeConverter();
            var schema = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object>
                {
                    ["name"] = new Dictionary<string, object> { ["type"] = "string" }
                }
            };

            var request = new UnifiedRequest
            {
                Model = "claude-opus-4-6",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Return structured data")
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.JsonSchema,
                    JsonSchema = new JsonSchema
                    {
                        Name = "person",
                        Schema = schema
                    }
                }
            };

            // Act
            var result = converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.OutputConfig);
            Assert.NotNull(result.OutputConfig.Format);
            Assert.Equal("json_schema", result.OutputConfig.Format.Type);
            Assert.NotNull(result.OutputConfig.Format.Schema);
            Assert.Equal(schema, result.OutputConfig.Format.Schema);
        }

        [Fact]
        public void Gemini_StructuredOutput_JsonMode_SetsResponseMimeType()
        {
            // Arrange
            var converter = new GeminiConverter();
            var request = new UnifiedRequest
            {
                Model = "gemini-3-pro",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Return JSON")
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.Json
                }
            };

            // Act
            var result = converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.GenerationConfig);
            Assert.Equal("application/json", result.GenerationConfig!.ResponseMimeType);
        }

        [Fact]
        public void Gemini_StructuredOutput_JsonSchema_SetsResponseSchema()
        {
            // Arrange
            var converter = new GeminiConverter();
            var schema = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object>
                {
                    ["name"] = new Dictionary<string, object> { ["type"] = "string" }
                }
            };

            var request = new UnifiedRequest
            {
                Model = "gemini-3-pro",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Return structured data")
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.JsonSchema,
                    JsonSchema = new JsonSchema
                    {
                        Name = "person",
                        Schema = schema
                    }
                }
            };

            // Act
            var result = converter.ConvertRequest(request);

            // Assert
            Assert.NotNull(result.GenerationConfig);
            Assert.Equal("application/json", result.GenerationConfig!.ResponseMimeType);
            Assert.NotNull(result.GenerationConfig.ResponseSchema);
            Assert.Equal(schema, result.GenerationConfig.ResponseSchema);
        }

        [Fact]
        public void ResponseFormat_DefaultValues_AreCorrect()
        {
            // Arrange & Act
            var format = new ResponseFormat
            {
                Type = ResponseFormatType.JsonSchema,
                JsonSchema = new JsonSchema
                {
                    Name = "test",
                    Schema = new Dictionary<string, object>()
                }
            };

            // Assert
            Assert.Equal(ResponseFormatType.JsonSchema, format.Type);
            Assert.True(format.JsonSchema.Strict); // Default is true
        }
    }
}
