using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Examples.ConsoleApp
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("LLM Abstraction Layer - Console Example");
            Console.WriteLine("========================================\n");

            // Get API keys from environment variables or user input
            var provider = GetProvider();
            var apiKey = GetApiKey(provider);

            if (string.IsNullOrEmpty(apiKey))
            {
                Console.WriteLine("No API key provided. Exiting...");
                return;
            }

            // Create the service
            var service = CreateService(provider, apiKey);
            var model = GetModelForProvider(provider);

            Console.WriteLine($"\nUsing {provider} with model {model}");
            Console.WriteLine("Type your messages (or 'quit' to exit)\n");

            // Conversation history
            var messages = new List<UnifiedMessage>();

            while (true)
            {
                Console.Write("You: ");
                var userInput = Console.ReadLine();

                if (string.IsNullOrEmpty(userInput) || userInput.ToLower() == "quit")
                    break;

                // Add user message
                messages.Add(new UnifiedMessage(MessageRole.User, userInput));

                // Create request
                var request = new UnifiedRequest
                {
                    Model = model,
                    Messages = messages,
                    System = "You are a helpful assistant.",
                    Parameters = new GenerationParameters
                    {
                        MaxTokens = 500,
                        Temperature = 0.7
                    }
                };

                try
                {
                    // Generate response
                    var response = await service.GenerateAsync(request);
                    var choice = response.Choices[0];
                    var textContent = choice.Message.Content[0] as TextContent;

                    if (textContent != null)
                    {
                        Console.WriteLine($"\nAssistant: {textContent.Text}\n");
                        Console.WriteLine($"[Tokens: {response.Usage.TotalTokens}, " +
                            $"Finish: {choice.FinishReason}]\n");

                        // Add assistant message to history
                        messages.Add(choice.Message);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nError: {ex.Message}\n");
                }
            }

            Console.WriteLine("\nGoodbye!");
        }

        static LLMProvider GetProvider()
        {
            Console.WriteLine("Select a provider:");
            Console.WriteLine("1. OpenAI");
            Console.WriteLine("2. Claude");
            Console.WriteLine("3. Gemini");
            Console.Write("\nChoice (1-3): ");

            var choice = Console.ReadLine();
            return choice switch
            {
                "1" => LLMProvider.OpenAI,
                "2" => LLMProvider.Claude,
                "3" => LLMProvider.Gemini,
                _ => LLMProvider.OpenAI
            };
        }

        static string GetApiKey(LLMProvider provider)
        {
            var envVar = provider switch
            {
                LLMProvider.OpenAI => "OPENAI_API_KEY",
                LLMProvider.Claude => "CLAUDE_API_KEY",
                LLMProvider.Gemini => "GEMINI_API_KEY",
                _ => ""
            };

            var apiKey = Environment.GetEnvironmentVariable(envVar);

            if (string.IsNullOrEmpty(apiKey))
            {
                Console.Write($"\nEnter your {provider} API key: ");
                apiKey = Console.ReadLine();
            }

            return apiKey ?? string.Empty;
        }

        static ILLMService CreateService(LLMProvider provider, string apiKey)
        {
            return provider switch
            {
                LLMProvider.OpenAI => LLMServiceFactory.CreateOpenAI(apiKey),
                LLMProvider.Claude => LLMServiceFactory.CreateClaude(apiKey),
                LLMProvider.Gemini => LLMServiceFactory.CreateGemini(apiKey),
                _ => throw new ArgumentException($"Unknown provider: {provider}")
            };
        }

        static string GetModelForProvider(LLMProvider provider)
        {
            return provider switch
            {
                LLMProvider.OpenAI => "gpt-4o-mini",
                LLMProvider.Claude => "claude-opus-4-6",
                LLMProvider.Gemini => "gemini-2.5-flash",
                _ => throw new ArgumentException($"Unknown provider: {provider}")
            };
        }
    }
}
