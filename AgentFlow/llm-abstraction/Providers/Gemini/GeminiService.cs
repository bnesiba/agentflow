using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini.Models;

namespace LLMAbstraction.Providers.Gemini
{
    /// <summary>
    /// Service for interacting with Gemini API
    /// </summary>
    public class GeminiService : ILLMService
    {
        private readonly HttpClient _httpClient;
        private readonly GeminiConverter _converter;
        private readonly string _apiKey;
        private readonly string _baseUrl;

        public GeminiService(string apiKey, string? baseUrl = null)
        {
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _baseUrl = baseUrl ?? "https://generativelanguage.googleapis.com/v1beta";
            _converter = new GeminiConverter();
            
            _httpClient = new HttpClient
            {
                BaseAddress = new Uri(_baseUrl)
            };
        }

        public GeminiService(
            HttpClient httpClient, 
            string apiKey,
            GeminiConverter? converter = null)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _baseUrl = httpClient.BaseAddress?.ToString() ?? 
                "https://generativelanguage.googleapis.com/v1beta";
            _converter = converter ?? new GeminiConverter();
        }

        public async Task<UnifiedResponse> GenerateAsync(
            UnifiedRequest request, 
            CancellationToken cancellationToken = default)
        {
            // Convert unified request to Gemini format
            var geminiRequest = _converter.ConvertRequest(request);

            // Build endpoint URL with model and API key
            // Gemini uses model in the URL path
            var endpoint = $"/models/{request.Model}:generateContent?key={_apiKey}";

            // Serialize request
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            var jsonContent = JsonSerializer.Serialize(geminiRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            // Make API call
            var response = await _httpClient.PostAsync(
                endpoint, 
                content, 
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException(
                    $"Gemini API request failed with status {response.StatusCode}: {errorContent}");
            }

            // Parse response
            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            var geminiResponse = JsonSerializer.Deserialize<GeminiGenerateResponse>(
                responseJson, 
                jsonOptions);

            if (geminiResponse == null)
            {
                throw new InvalidOperationException("Failed to deserialize Gemini response");
            }

            // Convert to unified format
            return _converter.ConvertResponse(geminiResponse);
        }

        public async IAsyncEnumerable<StreamChunk> StreamAsync(
            UnifiedRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // Convert unified request to Gemini format
            var geminiRequest = _converter.ConvertRequest(request);

            // Serialize request
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            var jsonContent = JsonSerializer.Serialize(geminiRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            // Use streaming endpoint
            var endpoint = $"/models/{request.Model}:streamGenerateContent?alt=sse&key={_apiKey}";
            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, endpoint)
            {
                Content = content
            };
            var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException(
                    $"Gemini API request failed with status {response.StatusCode}: {errorContent}");
            }

            // Stream response as server-sent events.
            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);

            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data: "))
                    continue;

                var data = line.Substring(6);
                GeminiStreamChunk? chunk;
                try
                {
                    chunk = JsonSerializer.Deserialize<GeminiStreamChunk>(data, jsonOptions);
                }
                catch
                {
                    continue;
                }

                if (chunk?.Candidates == null || chunk.Candidates.Count == 0)
                    continue;

                // Convert to unified format
                foreach (var candidate in chunk.Candidates)
                {
                    yield return ConvertStreamChunk(request.Model, chunk, candidate);
                }
            }
        }

        private StreamChunk ConvertStreamChunk(string model, GeminiStreamChunk chunk, GeminiCandidate candidate)
        {
            var delta = new StreamDelta();

            // Extract text content
            if (candidate.Content?.Parts != null)
            {
                foreach (var part in candidate.Content.Parts)
                {
                    if (!string.IsNullOrEmpty(part.Text))
                    {
                        delta.Content = part.Text;
                        break; // Take first text part
                    }
                }
            }

            return new StreamChunk
            {
                Id = chunk.ModelVersion ?? Guid.NewGuid().ToString(),
                Model = model,
                ChoiceIndex = candidate.Index,
                Delta = delta,
                FinishReason = candidate.FinishReason switch
                {
                    "STOP" => FinishReason.Stop,
                    "MAX_TOKENS" => FinishReason.MaxTokens,
                    "SAFETY" => FinishReason.ContentFilter,
                    "RECITATION" => FinishReason.ContentFilter,
                    _ => null
                },
                Usage = chunk.UsageMetadata != null ? new UsageInfo
                {
                    InputTokens = chunk.UsageMetadata.PromptTokenCount,
                    OutputTokens = chunk.UsageMetadata.CandidatesTokenCount,
                    TotalTokens = chunk.UsageMetadata.TotalTokenCount,
                    ReasoningTokens = chunk.UsageMetadata.ThoughtsTokenCount
                } : null
            };
        }
    }
}
