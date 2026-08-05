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
using LLMAbstraction.Core.Errors;
using LLMAbstraction.Core;
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
                BaseAddress = CreateBaseAddress(_baseUrl)
            };
            ConfigureHeaders();
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
            _httpClient.BaseAddress = CreateBaseAddress(_baseUrl);
            ConfigureHeaders();
        }

        public async Task<UnifiedResponse> GenerateAsync(
            UnifiedRequest request, 
            CancellationToken cancellationToken = default)
        {
            // Convert unified request to Gemini format
            var geminiRequest = _converter.ConvertRequest(request);

            // Build endpoint URL with model and API key
            // Gemini uses model in the URL path
            var endpoint = $"models/{request.Model}:generateContent";

            // Serialize request
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            var jsonContent = JsonSerializer.Serialize(geminiRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            // Make API call
            using var response = await _httpClient.PostAsync(
                endpoint, 
                content, 
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(LLMProvider.Gemini, response, errorContent);
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
            var endpoint = $"models/{request.Model}:streamGenerateContent?alt=sse";
            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, endpoint)
            {
                Content = content
            };
            using var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(LLMProvider.Gemini, response, errorContent);
            }

            // Stream response as server-sent events.
            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);
            var accumulatedMessages = new Dictionary<int, UnifiedMessage>();

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
                catch (JsonException exception)
                {
                    throw new InvalidOperationException(
                        "Failed to deserialize a Gemini streaming response event.",
                        exception);
                }

                if (chunk == null)
                    continue;

                if (chunk.Candidates == null || chunk.Candidates.Count == 0)
                {
                    if (chunk.UsageMetadata != null)
                    {
                        yield return new StreamChunk
                        {
                            Id = chunk.ResponseId ?? string.Empty,
                            Model = chunk.ModelVersion ?? request.Model,
                            ChoiceIndex = 0,
                            Delta = new StreamDelta(),
                            Usage = ConvertUsage(chunk.UsageMetadata)
                        };
                    }
                    continue;
                }

                // Convert to unified format
                foreach (var candidate in chunk.Candidates)
                {
                    if (!accumulatedMessages.TryGetValue(candidate.Index, out var accumulatedMessage))
                    {
                        accumulatedMessage = new UnifiedMessage
                        {
                            Role = MessageRole.Assistant,
                            Content = new List<ContentBlock>()
                        };
                        accumulatedMessages[candidate.Index] = accumulatedMessage;
                    }

                    yield return ConvertStreamChunk(request.Model, chunk, candidate, accumulatedMessage);
                }
            }
        }

        private StreamChunk ConvertStreamChunk(
            string requestedModel,
            GeminiStreamChunk chunk,
            GeminiCandidate candidate,
            UnifiedMessage accumulatedMessage)
        {
            var delta = new StreamDelta
            {
                ContentBlocks = new List<ContentBlock>(),
                ToolCalls = new List<ToolCallDelta>()
            };
            var text = new StringBuilder();

            // Extract text content
            if (candidate.Content?.Parts != null)
            {
                foreach (var part in candidate.Content.Parts)
                {
                    var contentBlock = GeminiConverter.ConvertResponsePart(part);
                    delta.ContentBlocks.Add(contentBlock);
                    accumulatedMessage.Content.Add(contentBlock);

                    if (contentBlock is TextContent textContent)
                    {
                        text.Append(textContent.Text);
                    }
                    else if (contentBlock is ToolCallContent toolCall)
                    {
                        delta.ToolCalls.Add(new ToolCallDelta
                        {
                            Index = delta.ToolCalls.Count,
                            Id = toolCall.Id,
                            Name = toolCall.Name,
                            CompleteArguments = JsonSerializer.Serialize(toolCall.Input),
                            IsComplete = true,
                            Type = "function",
                            NativeRepresentation = toolCall.NativeRepresentation
                        });
                    }
                }
            }

            delta.Content = text.Length > 0 ? text.ToString() : null;
            if (delta.ContentBlocks.Count == 0)
                delta.ContentBlocks = null;
            if (delta.ToolCalls.Count == 0)
                delta.ToolCalls = null;

            var finishReason = delta.ToolCalls != null
                ? FinishReason.ToolCalls
                : ConvertFinishReason(candidate.FinishReason);

            return new StreamChunk
            {
                Id = chunk.ResponseId ?? string.Empty,
                Model = chunk.ModelVersion ?? requestedModel,
                ChoiceIndex = candidate.Index,
                Delta = delta,
                FinishReason = finishReason,
                Usage = chunk.UsageMetadata != null ? ConvertUsage(chunk.UsageMetadata) : null,
                CompletedMessage = candidate.FinishReason != null ? accumulatedMessage : null
            };
        }

        private static FinishReason? ConvertFinishReason(string? finishReason)
        {
            return finishReason switch
            {
                "STOP" => FinishReason.Stop,
                "MAX_TOKENS" => FinishReason.MaxTokens,
                "SAFETY" => FinishReason.ContentFilter,
                "RECITATION" => FinishReason.ContentFilter,
                "BLOCKLIST" => FinishReason.ContentFilter,
                "PROHIBITED_CONTENT" => FinishReason.ContentFilter,
                "SPII" => FinishReason.ContentFilter,
                "IMAGE_SAFETY" => FinishReason.ContentFilter,
                "MALFORMED_FUNCTION_CALL" => FinishReason.Error,
                "UNEXPECTED_TOOL_CALL" => FinishReason.Error,
                null => null,
                _ => FinishReason.Other
            };
        }

        private static UsageInfo ConvertUsage(GeminiUsageMetadata usage)
        {
            return new UsageInfo
            {
                InputTokens = usage.PromptTokenCount,
                OutputTokens = usage.CandidatesTokenCount,
                TotalTokens = usage.TotalTokenCount,
                ReasoningTokens = usage.ThoughtsTokenCount
            };
        }

        private static Uri CreateBaseAddress(string baseUrl)
        {
            return new Uri(baseUrl.EndsWith("/", StringComparison.Ordinal)
                ? baseUrl
                : baseUrl + "/");
        }

        private void ConfigureHeaders()
        {
            if (!_httpClient.DefaultRequestHeaders.Contains("x-goog-api-key"))
                _httpClient.DefaultRequestHeaders.TryAddWithoutValidation("x-goog-api-key", _apiKey);
        }
    }
}
