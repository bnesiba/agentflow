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
            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, "/v1/messages")
            {
                Content = content
            };
            AddRequestSpecificHeaders(httpRequest, request);
            var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
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
            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, "/v1/messages")
            {
                Content = content
            };
            AddRequestSpecificHeaders(httpRequest, request);
            var response = await _httpClient.SendAsync(
                httpRequest,
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
            FinishReason? finishReason = null;
            var contentBlocks = new Dictionary<int, ClaudeContentBlock>();

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

                        case "content_block_start":
                            var blockStart = JsonSerializer.Deserialize<ClaudeContentBlockStart>(data, jsonOptions);
                            if (blockStart?.ContentBlock != null)
                            {
                                contentBlocks[blockStart.Index] = blockStart.ContentBlock;

                                if (blockStart.ContentBlock.Type == "tool_use")
                                {
                                    yield return new StreamChunk
                                    {
                                        Id = messageId ?? string.Empty,
                                        Model = model ?? string.Empty,
                                        ChoiceIndex = 0,
                                        Delta = new StreamDelta
                                        {
                                            ToolCalls = new List<ToolCallDelta>
                                            {
                                                new ToolCallDelta
                                                {
                                                    Index = blockStart.Index,
                                                    Id = blockStart.ContentBlock.Id,
                                                    Name = blockStart.ContentBlock.Name,
                                                    Type = "tool_use"
                                                }
                                            }
                                        }
                                    };
                                }
                            }
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
                            else if (delta?.Delta?.PartialJson != null)
                            {
                                contentBlocks.TryGetValue(delta.Index, out var block);
                                yield return new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = new StreamDelta
                                    {
                                        ToolCalls = new List<ToolCallDelta>
                                        {
                                            new ToolCallDelta
                                            {
                                                Index = delta.Index,
                                                Id = block?.Id,
                                                Name = block?.Name,
                                                Arguments = delta.Delta.PartialJson,
                                                Type = "tool_use"
                                            }
                                        }
                                    }
                                };
                            }
                            else if (delta?.Delta?.Thinking != null || delta?.Delta?.Signature != null)
                            {
                                yield return new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = new StreamDelta(),
                                    ProviderMetadata = new Dictionary<string, object>
                                    {
                                        { "claude.deltaType", delta.Delta.Type },
                                        { "claude.thinking", delta.Delta.Thinking ?? string.Empty },
                                        { "claude.signature", delta.Delta.Signature ?? string.Empty }
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
                                        InputTokens = messageDelta.Usage.InputTokens,
                                        OutputTokens = messageDelta.Usage.OutputTokens,
                                        TotalTokens = messageDelta.Usage.InputTokens + messageDelta.Usage.OutputTokens
                                    } : null
                                };
                            }
                            break;
                    }
                }
            }
        }

        private static void AddRequestSpecificHeaders(HttpRequestMessage httpRequest, UnifiedRequest request)
        {
            if (request.ProviderOptions?.Claude != null &&
                request.ProviderOptions.Claude.TryGetValue("anthropicBeta", out var betaValue))
            {
                if (betaValue is string betaHeader && !string.IsNullOrWhiteSpace(betaHeader))
                {
                    httpRequest.Headers.TryAddWithoutValidation("anthropic-beta", betaHeader);
                    return;
                }
            }

            if (UsesClaudeFileId(request.Messages))
            {
                httpRequest.Headers.TryAddWithoutValidation("anthropic-beta", "files-api-2025-04-14");
            }
        }

        private static bool UsesClaudeFileId(List<UnifiedMessage> messages)
        {
            foreach (var message in messages)
            {
                foreach (var block in message.Content)
                {
                    if (block is MediaContent media &&
                        !string.IsNullOrEmpty(media.Source.FileId))
                    {
                        return true;
                    }
                }
            }

            return false;
        }
    }
}
