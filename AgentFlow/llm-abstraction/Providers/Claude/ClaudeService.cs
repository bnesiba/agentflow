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
using LLMAbstraction.Providers.Claude.Models;

namespace LLMAbstraction.Providers.Claude
{
    /// <summary>
    /// Service for interacting with Claude API
    /// </summary>
    public class ClaudeService : ILLMService
    {
        private readonly HttpClient _httpClient;
        private readonly ClaudeConverter _converter;
        private readonly string _apiKey;
        private readonly string _apiVersion;
        private readonly string _baseUrl;

        public ClaudeService(string apiKey, string? apiVersion = null, string? baseUrl = null)
        {
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _apiVersion = apiVersion ?? "2023-06-01";
            _baseUrl = baseUrl ?? "https://api.anthropic.com";
            _converter = new ClaudeConverter();
            
            _httpClient = new HttpClient
            {
                BaseAddress = new Uri(_baseUrl)
            };
            ConfigureHeaders();
        }

        public ClaudeService(
            HttpClient httpClient, 
            string apiKey, 
            string? apiVersion = null,
            ClaudeConverter? converter = null)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _apiVersion = apiVersion ?? "2023-06-01";
            _baseUrl = httpClient.BaseAddress?.ToString() ?? "https://api.anthropic.com";
            _converter = converter ?? new ClaudeConverter();

            ConfigureHeaders();
        }

        private void ConfigureHeaders()
        {
            _httpClient.DefaultRequestHeaders.Clear();
            _httpClient.DefaultRequestHeaders.Add("x-api-key", _apiKey);
            _httpClient.DefaultRequestHeaders.Add("anthropic-version", _apiVersion);
            _httpClient.DefaultRequestHeaders.Add("content-type", "application/json");
        }

        public async Task<UnifiedResponse> GenerateAsync(
            UnifiedRequest request, 
            CancellationToken cancellationToken = default)
        {
            // Convert unified request to Claude format
            var claudeRequest = _converter.ConvertRequest(request);

            // Serialize request
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            var jsonContent = JsonSerializer.Serialize(claudeRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            // Make API call
            var response = await _httpClient.PostAsync(
                "/v1/messages", 
                content, 
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException(
                    $"Claude API request failed with status {response.StatusCode}: {errorContent}");
            }

            // Parse response
            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            var claudeResponse = JsonSerializer.Deserialize<ClaudeMessageResponse>(
                responseJson, 
                jsonOptions);

            if (claudeResponse == null)
            {
                throw new InvalidOperationException("Failed to deserialize Claude response");
            }

            // Convert to unified format
            return _converter.ConvertResponse(claudeResponse);
        }

        public async IAsyncEnumerable<StreamChunk> StreamAsync(
            UnifiedRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            // Convert unified request to Claude format and enable streaming
            var claudeRequest = _converter.ConvertRequest(request);
            claudeRequest.Stream = true;

            // Serialize request
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            var jsonContent = JsonSerializer.Serialize(claudeRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            // Make API call
            var response = await _httpClient.PostAsync(
                "/v1/messages",
                content,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException(
                    $"Claude API request failed with status {response.StatusCode}: {errorContent}");
            }

            // Stream response
            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);

            string? messageId = null;
            string? model = null;
            string? currentContent = null;
            FinishReason? finishReason = null;

            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("event: ") && !line.StartsWith("data: "))
                    continue;

                if (line.StartsWith("event: "))
                {
                    var eventType = line.Substring(7);
                    var dataLine = await reader.ReadLineAsync();
                    if (dataLine == null || !dataLine.StartsWith("data: "))
                        continue;

                    var data = dataLine.Substring(6);
                    
                    switch (eventType)
                    {
                        case "message_start":
                            var messageStart = JsonSerializer.Deserialize<ClaudeMessageStart>(data, jsonOptions);
                            messageId = messageStart?.Message?.Id;
                            model = messageStart?.Message?.Model;
                            break;

                        case "content_block_delta":
                            var delta = JsonSerializer.Deserialize<ClaudeContentBlockDelta>(data, jsonOptions);
                            if (delta?.Delta?.Text != null)
                            {
                                yield return new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = new StreamDelta
                                    {
                                        Content = delta.Delta.Text
                                    }
                                };
                            }
                            break;

                        case "message_delta":
                            var messageDelta = JsonSerializer.Deserialize<ClaudeMessageDelta>(data, jsonOptions);
                            if (messageDelta?.Delta?.StopReason != null)
                            {
                                finishReason = messageDelta.Delta.StopReason switch
                                {
                                    "end_turn" => FinishReason.Stop,
                                    "max_tokens" => FinishReason.MaxTokens,
                                    "stop_sequence" => FinishReason.Stop,
                                    "tool_use" => FinishReason.ToolCalls,
                                    _ => FinishReason.Other
                                };

                                yield return new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = new StreamDelta(),
                                    FinishReason = finishReason,
                                    Usage = messageDelta.Usage != null ? new UsageInfo
                                    {
                                        PromptTokens = messageDelta.Usage.InputTokens,
                                        CompletionTokens = messageDelta.Usage.OutputTokens,
                                        TotalTokens = messageDelta.Usage.InputTokens + messageDelta.Usage.OutputTokens
                                    } : null
                                };
                            }
                            break;
                    }
                }
            }
        }
    }
}
