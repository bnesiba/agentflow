using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.OpenAI.Models;

namespace LLMAbstraction.Providers.OpenAI
{
    /// <summary>
    /// Service for interacting with OpenAI API
    /// </summary>
    public class OpenAIService : ILLMService
    {
        private readonly HttpClient _httpClient;
        private readonly OpenAIConverter _converter;
        private readonly string _apiKey;
        private readonly string _baseUrl;

        public OpenAIService(string apiKey, string? baseUrl = null)
        {
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _baseUrl = baseUrl ?? "https://api.openai.com/v1";
            _converter = new OpenAIConverter();
            
            _httpClient = new HttpClient
            {
                BaseAddress = new Uri(_baseUrl)
            };
            _httpClient.DefaultRequestHeaders.Authorization = 
                new AuthenticationHeaderValue("Bearer", _apiKey);
        }

        public OpenAIService(HttpClient httpClient, string apiKey, OpenAIConverter? converter = null)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _baseUrl = httpClient.BaseAddress?.ToString() ?? "https://api.openai.com/v1";
            _converter = converter ?? new OpenAIConverter();

            if (_httpClient.DefaultRequestHeaders.Authorization == null)
            {
                _httpClient.DefaultRequestHeaders.Authorization = 
                    new AuthenticationHeaderValue("Bearer", _apiKey);
            }
        }

        public async Task<UnifiedResponse> GenerateAsync(
            UnifiedRequest request, 
            CancellationToken cancellationToken = default)
        {
            // Convert unified request to OpenAI format
            var openAIRequest = _converter.ConvertRequest(request);

            // Serialize request
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            var jsonContent = JsonSerializer.Serialize(openAIRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            // Make API call
            var response = await _httpClient.PostAsync(
                "/chat/completions", 
                content, 
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException(
                    $"OpenAI API request failed with status {response.StatusCode}: {errorContent}");
            }

            // Parse response
            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            var openAIResponse = JsonSerializer.Deserialize<OpenAIChatResponse>(
                responseJson, 
                jsonOptions);

            if (openAIResponse == null)
            {
                throw new InvalidOperationException("Failed to deserialize OpenAI response");
            }

            // Convert to unified format
            return _converter.ConvertResponse(openAIResponse);
        }

        public async IAsyncEnumerable<StreamChunk> StreamAsync(
            UnifiedRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // Convert unified request to OpenAI format and enable streaming
            var openAIRequest = _converter.ConvertRequest(request);
            openAIRequest.Stream = true;

            // Serialize request
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            var jsonContent = JsonSerializer.Serialize(openAIRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            // Make API call
            var response = await _httpClient.PostAsync(
                "/chat/completions",
                content,
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException(
                    $"OpenAI API request failed with status {response.StatusCode}: {errorContent}");
            }

            // Stream response
            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);

            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data: "))
                    continue;

                var data = line.Substring(6); // Remove "data: " prefix
                if (data == "[DONE]")
                    break;

                OpenAIStreamChunk? chunk;
                try
                {
                    chunk = JsonSerializer.Deserialize<OpenAIStreamChunk>(data, jsonOptions);
                }
                catch
                {
                    continue; // Skip malformed chunks
                }

                if (chunk?.Choices == null || chunk.Choices.Count == 0)
                    continue;

                // Convert to unified format
                foreach (var choice in chunk.Choices)
                {
                    yield return ConvertStreamChunk(chunk, choice);
                }
            }
        }

        private StreamChunk ConvertStreamChunk(OpenAIStreamChunk chunk, OpenAIStreamChoice choice)
        {
            var delta = new StreamDelta
            {
                Role = choice.Delta.Role switch
                {
                    "assistant" => MessageRole.Assistant,
                    "user" => MessageRole.User,
                    "system" => MessageRole.System,
                    _ => null
                },
                Content = choice.Delta.Content
            };

            // Convert tool calls if present
            if (choice.Delta.ToolCalls != null)
            {
                delta.ToolCalls = new List<ToolCallDelta>();
                foreach (var toolCall in choice.Delta.ToolCalls)
                {
                    delta.ToolCalls.Add(new ToolCallDelta
                    {
                        Index = toolCall.Index,
                        Id = toolCall.Id,
                        Name = toolCall.Function?.Name,
                        Arguments = toolCall.Function?.Arguments
                    });
                }
            }

            return new StreamChunk
            {
                Id = chunk.Id,
                Model = chunk.Model,
                ChoiceIndex = choice.Index,
                Delta = delta,
                FinishReason = choice.FinishReason switch
                {
                    "stop" => FinishReason.Stop,
                    "length" => FinishReason.MaxTokens,
                    "tool_calls" => FinishReason.ToolCalls,
                    "content_filter" => FinishReason.ContentFilter,
                    _ => null
                },
                Usage = chunk.Usage != null ? new UsageInfo
                {
                    PromptTokens = chunk.Usage.PromptTokens,
                    CompletionTokens = chunk.Usage.CompletionTokens,
                    TotalTokens = chunk.Usage.TotalTokens
                } : null
            };
        }
    }
}
