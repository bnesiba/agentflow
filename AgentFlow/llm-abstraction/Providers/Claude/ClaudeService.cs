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
            claudeRequest.Stream = false;

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
            using var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(LLMProvider.Claude, response, errorContent);
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
            using var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(LLMProvider.Claude, response, errorContent);
            }

            // Stream response
            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);

            string? messageId = null;
            string? model = null;
            FinishReason? finishReason = null;
            ClaudeUsage? initialUsage = null;
            var contentBlocks = new Dictionary<int, ClaudeStreamBlockState>();
            var completedMessage = new UnifiedMessage
            {
                Role = MessageRole.Assistant,
                Content = new List<ContentBlock>()
            };

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
                            initialUsage = messageStart?.Message?.Usage;
                            break;

                        case "content_block_start":
                            var blockStart = JsonSerializer.Deserialize<ClaudeContentBlockStart>(data, jsonOptions);
                            if (blockStart?.ContentBlock != null)
                            {
                                contentBlocks[blockStart.Index] = new ClaudeStreamBlockState(blockStart.ContentBlock);

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
                                if (contentBlocks.TryGetValue(delta.Index, out var textState))
                                    textState.Text.Append(delta.Delta.Text);

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
                                block?.PartialJson.Append(delta.Delta.PartialJson);
                                if (block?.Block.Type == "tool_use")
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
                                                    Index = delta.Index,
                                                    Id = block.Block.Id,
                                                    Name = block.Block.Name,
                                                    Arguments = delta.Delta.PartialJson,
                                                    Type = "tool_use"
                                                }
                                            }
                                        }
                                    };
                                }
                            }
                            else if (delta?.Delta?.Thinking != null || delta?.Delta?.Signature != null)
                            {
                                if (contentBlocks.TryGetValue(delta.Index, out var thinkingState))
                                {
                                    if (delta.Delta.Thinking != null)
                                        thinkingState.Thinking.Append(delta.Delta.Thinking);
                                    if (delta.Delta.Signature != null)
                                        thinkingState.Signature.Append(delta.Delta.Signature);
                                }

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

                        case "content_block_stop":
                            var blockStop = JsonSerializer.Deserialize<ClaudeContentBlockStop>(data, jsonOptions);
                            if (blockStop != null && contentBlocks.Remove(blockStop.Index, out var completedState))
                            {
                                var completedBlock = completedState.Complete();
                                var unifiedBlock = ClaudeConverter.ConvertResponseContentBlock(completedBlock);
                                completedMessage.Content.Add(unifiedBlock);

                                var completedDelta = new StreamDelta
                                {
                                    ContentBlocks = new List<ContentBlock> { unifiedBlock }
                                };

                                if (completedBlock.Type == "tool_use")
                                {
                                    completedDelta.ToolCalls = new List<ToolCallDelta>
                                    {
                                        new ToolCallDelta
                                        {
                                            Index = blockStop.Index,
                                            Id = completedBlock.Id,
                                            Name = completedBlock.Name,
                                            CompleteArguments = completedState.PartialJson.ToString(),
                                            IsComplete = true,
                                            Type = "tool_use",
                                            NativeRepresentation = unifiedBlock.NativeRepresentation
                                        }
                                    };
                                }

                                yield return new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = completedDelta
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
                                    "pause_turn" => FinishReason.Pause,
                                    "refusal" => FinishReason.ContentFilter,
                                    "model_context_window_exceeded" => FinishReason.MaxTokens,
                                    _ => FinishReason.Other
                                };

                                yield return new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = new StreamDelta(),
                                    FinishReason = finishReason,
                                    Usage = ConvertStreamingUsage(initialUsage, messageDelta.Usage),
                                    CompletedMessage = completedMessage
                                };
                            }
                            break;
                    }
                }
            }
        }

        private static UsageInfo? ConvertStreamingUsage(ClaudeUsage? initial, ClaudeUsage? final)
        {
            if (initial == null && final == null)
                return null;

            var inputTokens = initial?.InputTokens ?? final?.InputTokens ?? 0;
            var outputTokens = final?.OutputTokens ?? initial?.OutputTokens ?? 0;
            return new UsageInfo
            {
                InputTokens = inputTokens,
                OutputTokens = outputTokens,
                TotalTokens = inputTokens + outputTokens,
                CacheCreationTokens = initial?.CacheCreationInputTokens ?? final?.CacheCreationInputTokens,
                CacheReadTokens = initial?.CacheReadInputTokens ?? final?.CacheReadInputTokens,
                ProviderMetadata = (final?.ServerToolUse ?? initial?.ServerToolUse) is { } serverToolUse
                    ? new Dictionary<string, object> { ["claude.serverToolUse"] = serverToolUse }
                    : null
            };
        }

        private sealed class ClaudeStreamBlockState
        {
            public ClaudeStreamBlockState(ClaudeContentBlock block)
            {
                Block = block;
                Text.Append(block.Text);
                Thinking.Append(block.Thinking);
                Signature.Append(block.Signature);
                if (block.Input != null && block.Input.Count > 0)
                    PartialJson.Append(JsonSerializer.Serialize(block.Input));
            }

            public ClaudeContentBlock Block { get; }
            public StringBuilder Text { get; } = new();
            public StringBuilder Thinking { get; } = new();
            public StringBuilder Signature { get; } = new();
            public StringBuilder PartialJson { get; } = new();

            public ClaudeContentBlock Complete()
            {
                if (Block.Type == "text")
                    Block.Text = Text.ToString();
                else if (Block.Type == "thinking")
                {
                    Block.Thinking = Thinking.ToString();
                    Block.Signature = Signature.ToString();
                }
                else if (Block.Type is "tool_use" or "server_tool_use" && PartialJson.Length > 0)
                {
                    Block.Input = JsonSerializer.Deserialize<Dictionary<string, object>>(
                        PartialJson.ToString()) ?? new Dictionary<string, object>();
                }

                return Block;
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
