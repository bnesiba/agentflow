using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Examples
{
    /// <summary>
    /// Examples demonstrating basic usage of the LLM abstraction layer
    /// </summary>
    public class BasicUsage
    {
        /// <summary>
        /// Simple text generation example
        /// </summary>
        public static async Task SimpleTextGeneration()
        {
            // Create a service (can be OpenAI, Claude, or Gemini)
            var service = LLMServiceFactory.CreateOpenAI("your-api-key-here");

            // Create a simple request
            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "What is the capital of France?")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 100,
                    Temperature = 0.7
                }
            };

            // Generate response
            var response = await service.GenerateAsync(request);

            // Access the result
            var firstChoice = response.Choices[0];
            var textContent = firstChoice.Message.Content[0] as TextContent;
            Console.WriteLine($"Response: {textContent?.Text}");
            Console.WriteLine($"Tokens used: {response.Usage.TotalTokens}");
        }

        /// <summary>
        /// Multi-turn conversation example
        /// </summary>
        public static async Task MultiTurnConversation()
        {
            var service = LLMServiceFactory.CreateClaude("your-api-key-here");

            var request = new UnifiedRequest
            {
                Model = "claude-opus-4-6",
                System = "You are a helpful assistant that speaks like a pirate.",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Hello! Who are you?"),
                    new UnifiedMessage(MessageRole.Assistant, "Ahoy there, matey! I be a helpful assistant, ready to answer yer questions!"),
                    new UnifiedMessage(MessageRole.User, "What's the weather like today?")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 150,
                    Temperature = 0.8
                }
            };

            var response = await service.GenerateAsync(request);
            var textContent = response.Choices[0].Message.Content[0] as TextContent;
            Console.WriteLine($"Pirate Assistant: {textContent?.Text}");
        }

        /// <summary>
        /// Using the same code with different providers
        /// </summary>
        public static async Task ProviderAgnosticExample()
        {
            // Define the request once
            var request = new UnifiedRequest
            {
                Model = "placeholder", // Will be set per provider
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "Explain quantum computing in one sentence.")
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 100,
                    Temperature = 0.5
                }
            };

            // Try with OpenAI
            Console.WriteLine("=== OpenAI ===");
            var openAIService = LLMServiceFactory.CreateOpenAI("openai-key");
            request.Model = "gpt-4o";
            var openAIResponse = await openAIService.GenerateAsync(request);
            PrintResponse(openAIResponse);

            // Try with Claude
            Console.WriteLine("\n=== Claude ===");
            var claudeService = LLMServiceFactory.CreateClaude("claude-key");
            request.Model = "claude-opus-4-6";
            var claudeResponse = await claudeService.GenerateAsync(request);
            PrintResponse(claudeResponse);

            // Try with Gemini
            Console.WriteLine("\n=== Gemini ===");
            var geminiService = LLMServiceFactory.CreateGemini("gemini-key");
            request.Model = "gemini-3-pro";
            var geminiResponse = await geminiService.GenerateAsync(request);
            PrintResponse(geminiResponse);
        }

        /// <summary>
        /// Tool/function calling example
        /// </summary>
        public static async Task ToolCallingExample()
        {
            var service = LLMServiceFactory.CreateOpenAI("your-api-key-here");

            // Define a tool
            var weatherTool = new ToolDefinition
            {
                Name = "get_weather",
                Description = "Get the current weather for a location",
                Parameters = new Dictionary<string, object>
                {
                    ["type"] = "object",
                    ["properties"] = new Dictionary<string, object>
                    {
                        ["location"] = new Dictionary<string, object>
                        {
                            ["type"] = "string",
                            ["description"] = "The city and state, e.g. San Francisco, CA"
                        },
                        ["unit"] = new Dictionary<string, object>
                        {
                            ["type"] = "string",
                            ["enum"] = new[] { "celsius", "fahrenheit" }
                        }
                    },
                    ["required"] = new[] { "location" }
                }
            };

            var request = new UnifiedRequest
            {
                Model = "gpt-4o",
                Messages = new List<UnifiedMessage>
                {
                    new UnifiedMessage(MessageRole.User, "What's the weather in Boston?")
                },
                Tools = new List<ToolDefinition> { weatherTool },
                ToolChoice = new ToolChoice { Type = ToolChoiceType.Auto },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 200
                }
            };

            var response = await service.GenerateAsync(request);

            // Check if model wants to call a tool
            var choice = response.Choices[0];
            if (choice.FinishReason == FinishReason.ToolCalls)
            {
                foreach (var content in choice.Message.Content)
                {
                    if (content is ToolCallContent toolCall)
                    {
                        Console.WriteLine($"Model wants to call: {toolCall.Name}");
                        Console.WriteLine($"With arguments: {System.Text.Json.JsonSerializer.Serialize(toolCall.Input)}");

                        // In a real application, you would:
                        // 1. Execute the tool with the provided arguments
                        // 2. Add the tool result to the conversation
                        // 3. Send another request to get the final response
                    }
                }
            }
        }

        /// <summary>
        /// Multimodal example with image
        /// </summary>
        public static async Task MultimodalExample()
        {
            var service = LLMServiceFactory.CreateOpenAI("your-api-key-here");

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
                            Source = new ImageSource
                            {
                                Url = "https://example.com/image.jpg"
                            }
                        }
                    })
                },
                Parameters = new GenerationParameters
                {
                    MaxTokens = 300
                }
            };

            var response = await service.GenerateAsync(request);
            var textContent = response.Choices[0].Message.Content[0] as TextContent;
            Console.WriteLine($"Image description: {textContent?.Text}");
        }

        /// <summary>
        /// Helper method to print responses
        /// </summary>
        private static void PrintResponse(UnifiedResponse response)
        {
            var choice = response.Choices[0];
            var textContent = choice.Message.Content[0] as TextContent;
            Console.WriteLine($"Response: {textContent?.Text}");
            Console.WriteLine($"Finish Reason: {choice.FinishReason}");
            Console.WriteLine($"Tokens: {response.Usage.TotalTokens}");
        }
    }
}
