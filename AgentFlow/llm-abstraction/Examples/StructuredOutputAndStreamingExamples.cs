using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Examples
{
    /// <summary>
    /// Examples demonstrating structured output and streaming features
    /// </summary>
    public class StructuredOutputAndStreamingExamples
    {
        /// <summary>
        /// Example 1: Structured output with JSON mode
        /// </summary>
        public static async Task StructuredOutputJsonMode()
        {
            var service = LLMServiceFactory.CreateOpenAI("your-api-key");

            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.System, "You are a helpful assistant that outputs JSON."),
                    new UnifiedMessage(MessageRole.User, "List 3 colors with their hex codes.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 200,
                    Temperature = 0.7
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.Json
                }
            };

            var response = await service.GenerateAsync(request);
            var jsonContent = (response.Choices[0].Message.Content[0] as TextContent)?.Text;
            
            Console.WriteLine("JSON Response:");
            Console.WriteLine(jsonContent);
            // Output will be valid JSON like: {"colors": [{"name": "red", "hex": "#FF0000"}, ...]}
        }

        /// <summary>
        /// Example 2: Structured output with JSON Schema
        /// </summary>
        public static async Task StructuredOutputWithSchema()
        {
            var service = LLMServiceFactory.CreateOpenAI("your-api-key");

            // Define the schema for a person object
            var personSchema = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object>
                {
                    ["name"] = new Dictionary<string, object>
                    {
                        ["type"] = "string",
                        ["description"] = "The person's full name"
                    },
                    ["age"] = new Dictionary<string, object>
                    {
                        ["type"] = "number",
                        ["description"] = "The person's age in years"
                    },
                    ["email"] = new Dictionary<string, object>
                    {
                        ["type"] = "string",
                        ["format"] = "email",
                        ["description"] = "The person's email address"
                    },
                    ["interests"] = new Dictionary<string, object>
                    {
                        ["type"] = "array",
                        ["items"] = new Dictionary<string, object>
                        {
                            ["type"] = "string"
                        },
                        ["description"] = "List of the person's interests"
                    }
                },
                ["required"] = new[] { "name", "age", "email" },
                ["additionalProperties"] = false
            };

            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, 
                        "Generate a person profile for a software engineer named Alice who is 28 years old.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 300,
                    Temperature = 0.7
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.JsonSchema,
                    JsonSchema = new JsonSchema
                    {
                        Name = "person",
                        Description = "A person profile",
                        Schema = personSchema,
                        Strict = true  // Enforce strict schema validation
                    }
                }
            };

            var response = await service.GenerateAsync(request);
            var structuredData = (response.Choices[0].Message.Content[0] as TextContent)?.Text;
            
            Console.WriteLine("Structured Person Data:");
            Console.WriteLine(structuredData);
            // Output will strictly conform to the schema
        }

        /// <summary>
        /// Example 3: Basic streaming
        /// </summary>
        public static async Task BasicStreaming()
        {
            var service = LLMServiceFactory.CreateOpenAI("your-api-key");

            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Write a short poem about coding.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 200,
                    Temperature = 0.8
                }
            };

            Console.WriteLine("Streaming response:");
            
            await foreach (var chunk in service.StreamAsync(request))
            {
                if (!string.IsNullOrEmpty(chunk.Delta.Content))
                {
                    Console.Write(chunk.Delta.Content);
                }

                if (chunk.FinishReason.HasValue)
                {
                    Console.WriteLine($"\n\nFinished: {chunk.FinishReason}");
                    if (chunk.Usage != null)
                    {
                        Console.WriteLine($"Tokens used: {chunk.Usage.TotalTokens}");
                    }
                }
            }
        }

        /// <summary>
        /// Example 4: Streaming with aggregation
        /// </summary>
        public static async Task StreamingWithAggregation()
        {
            var service = LLMServiceFactory.CreateClaude("your-api-key");

            var request = new UnifiedRequest
            {
                Model = "claude-opus-4-6",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Explain quantum computing in simple terms.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 500,
                    Temperature = 0.7
                }
            };

            var fullResponse = new System.Text.StringBuilder();
            
            await foreach (var chunk in service.StreamAsync(request))
            {
                if (!string.IsNullOrEmpty(chunk.Delta.Content))
                {
                    fullResponse.Append(chunk.Delta.Content);
                    Console.Write(chunk.Delta.Content); // Stream to console
                }
            }

            Console.WriteLine("\n\n--- Full Response ---");
            Console.WriteLine(fullResponse.ToString());
        }

        /// <summary>
        /// Example 5: Structured output with Claude (native support)
        /// </summary>
        public static async Task ClaudeStructuredOutput()
        {
            var service = LLMServiceFactory.CreateClaude("your-api-key");

            var request = new UnifiedRequest
            {
                Model = "claude-opus-4-6",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "List 5 programming languages with their primary use cases.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 500,
                    Temperature = 0.5
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.Json
                }
                // Claude uses native output_config.format parameter
            };

            var response = await service.GenerateAsync(request);
            var jsonContent = (response.Choices[0].Message.Content[0] as TextContent)?.Text;
            
            Console.WriteLine("Claude JSON Response:");
            Console.WriteLine(jsonContent);
        }

        /// <summary>
        /// Example 6: Gemini structured output with schema
        /// </summary>
        public static async Task GeminiStructuredOutput()
        {
            var service = LLMServiceFactory.CreateGemini("your-api-key");

            var recipeSchema = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object>
                {
                    ["name"] = new Dictionary<string, object> { ["type"] = "string" },
                    ["ingredients"] = new Dictionary<string, object>
                    {
                        ["type"] = "array",
                        ["items"] = new Dictionary<string, object> { ["type"] = "string" }
                    },
                    ["steps"] = new Dictionary<string, object>
                    {
                        ["type"] = "array",
                        ["items"] = new Dictionary<string, object> { ["type"] = "string" }
                    },
                    ["prepTime"] = new Dictionary<string, object> { ["type"] = "number" }
                },
                ["required"] = new[] { "name", "ingredients", "steps" }
            };

            var request = new UnifiedRequest
            {
                Model = "gemini-2.5-flash",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Give me a recipe for chocolate chip cookies.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 1000,
                    Temperature = 0.7
                },
                ResponseFormat = new ResponseFormat
                {
                    Type = ResponseFormatType.JsonSchema,
                    JsonSchema = new JsonSchema
                    {
                        Name = "recipe",
                        Schema = recipeSchema
                    }
                }
            };

            var response = await service.GenerateAsync(request);
            var recipeJson = (response.Choices[0].Message.Content[0] as TextContent)?.Text;
            
            Console.WriteLine("Gemini Recipe (JSON):");
            Console.WriteLine(recipeJson);
        }

        /// <summary>
        /// Example 7: Streaming with tool calls
        /// </summary>
        public static async Task StreamingWithToolCalls()
        {
            var service = LLMServiceFactory.CreateOpenAI("your-api-key");

            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "What's the weather in Boston?")
                },
                Tools = new List<ToolDefinition>
                {
                    new ToolDefinition
                    {
                        Name = "get_weather",
                        Description = "Get the current weather in a location",
                        Parameters = new Dictionary<string, object>
                        {
                            ["type"] = "object",
                            ["properties"] = new Dictionary<string, object>
                            {
                                ["location"] = new Dictionary<string, object>
                                {
                                    ["type"] = "string",
                                    ["description"] = "The city and state"
                                }
                            },
                            ["required"] = new[] { "location" }
                        }
                    }
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 200
                }
            };

            var toolCallArgs = new System.Text.StringBuilder();
            
            await foreach (var chunk in service.StreamAsync(request))
            {
                if (chunk.Delta.ToolCalls != null)
                {
                    foreach (var toolCall in chunk.Delta.ToolCalls)
                    {
                        if (!string.IsNullOrEmpty(toolCall.Id))
                        {
                            Console.WriteLine($"Tool Call ID: {toolCall.Id}");
                        }
                        if (!string.IsNullOrEmpty(toolCall.Name))
                        {
                            Console.WriteLine($"Tool Name: {toolCall.Name}");
                        }
                        if (!string.IsNullOrEmpty(toolCall.Arguments))
                        {
                            toolCallArgs.Append(toolCall.Arguments);
                        }
                    }
                }

                if (chunk.FinishReason == FinishReason.ToolCalls)
                {
                    Console.WriteLine($"Tool Arguments: {toolCallArgs}");
                }
            }
        }
    }
}
