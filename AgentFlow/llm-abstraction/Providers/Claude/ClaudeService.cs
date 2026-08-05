using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Errors;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Transport;
using LLMAbstraction.Providers.Claude.Models;

namespace LLMAbstraction.Providers.Claude
{
    /// <summary>
    /// Service for interacting with Claude API
    /// </summary>
    public class ClaudeService : ILLMServiceWithTokenCounting
    {
        private readonly HttpClient _httpClient;
        private readonly ClaudeConverter _converter;
        private readonly string _apiKey;
        private readonly string _apiVersion;
        private readonly string _baseUrl;
        private readonly ProviderHttpTransport _transport;

        public ClaudeService(
            string apiKey,
            string? apiVersion = null,
            string? baseUrl = null,
            LLMTransportOptions? transportOptions = null)
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
            _transport = new ProviderHttpTransport(_httpClient, LLMProvider.Claude, transportOptions);
        }

        public ClaudeService(
            HttpClient httpClient, 
            string apiKey, 
            string? apiVersion = null,
            ClaudeConverter? converter = null,
            LLMTransportOptions? transportOptions = null)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _apiVersion = apiVersion ?? "2023-06-01";
            _baseUrl = httpClient.BaseAddress?.ToString() ?? "https://api.anthropic.com";
            _converter = converter ?? new ClaudeConverter();

            ConfigureHeaders();
            _transport = new ProviderHttpTransport(_httpClient, LLMProvider.Claude, transportOptions);
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
            using var transportResponse = await _transport.SendAsync(
                () => CreateRequest("/v1/messages", jsonContent, request),
                "/v1/messages",
                request.Model,
                false,
                cancellationToken,
                request.Transport);
            var response = transportResponse.Response;

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(
                    LLMProvider.Claude,
                    response,
                    errorContent,
                    transportResponse.Metadata);
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
            var converted = _converter.ConvertResponse(claudeResponse);
            converted.Transport = transportResponse.Metadata;
            return converted;
        }

        public async Task<TokenCountResult> CountInputTokensAsync(
            UnifiedRequest request,
            CancellationToken cancellationToken = default)
        {
            var claudeRequest = _converter.ConvertRequest(request);
            var jsonOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
            var jsonContent = CreateTokenCountJson(claudeRequest, jsonOptions);
            using var transportResponse = await _transport.SendAsync(
                () => CreateRequest("/v1/messages/count_tokens", jsonContent, request),
                "/v1/messages/count_tokens",
                request.Model,
                false,
                cancellationToken,
                request.Transport);
            var response = transportResponse.Response;

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(
                    LLMProvider.Claude,
                    response,
                    errorContent,
                    transportResponse.Metadata);
            }

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            var count = JsonSerializer.Deserialize<ClaudeMessageTokensCount>(responseJson, jsonOptions)
                ?? throw new InvalidOperationException("Failed to deserialize Claude token count response");

            return new TokenCountResult
            {
                Provider = LLMProvider.Claude,
                Model = request.Model,
                InputTokens = count.InputTokens,
                Accuracy = TokenCountAccuracy.Estimate,
                Transport = transportResponse.Metadata
            };
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
            using var transportResponse = await _transport.SendAsync(
                () => CreateRequest("/v1/messages", jsonContent, request),
                "/v1/messages",
                request.Model,
                true,
                cancellationToken,
                request.Transport);
            var response = transportResponse.Response;

            // Handle errors
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(
                    LLMProvider.Claude,
                    response,
                    errorContent,
                    transportResponse.Metadata);
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
            var transportAttached = false;

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
                                    yield return AttachTransport(new StreamChunk
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
                                    }, transportResponse.Metadata, ref transportAttached);
                                }
                            }
                            break;

                        case "content_block_delta":
                            var delta = JsonSerializer.Deserialize<ClaudeContentBlockDelta>(data, jsonOptions);
                            if (delta?.Delta?.Text != null)
                            {
                                if (contentBlocks.TryGetValue(delta.Index, out var textState))
                                    textState.Text.Append(delta.Delta.Text);

                                yield return AttachTransport(new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = new StreamDelta
                                    {
                                        Content = delta.Delta.Text
                                    }
                                }, transportResponse.Metadata, ref transportAttached);
                            }
                            else if (delta?.Delta?.PartialJson != null)
                            {
                                contentBlocks.TryGetValue(delta.Index, out var block);
                                block?.PartialJson.Append(delta.Delta.PartialJson);
                                if (block?.Block.Type == "tool_use")
                                {
                                    yield return AttachTransport(new StreamChunk
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
                                    }, transportResponse.Metadata, ref transportAttached);
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

                                yield return AttachTransport(new StreamChunk
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
                                }, transportResponse.Metadata, ref transportAttached);
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

                                yield return AttachTransport(new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = completedDelta
                                }, transportResponse.Metadata, ref transportAttached);
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

                                EvidenceProjector.Project(completedMessage, ProviderIds.Claude);

                                yield return AttachTransport(new StreamChunk
                                {
                                    Id = messageId ?? string.Empty,
                                    Model = model ?? string.Empty,
                                    ChoiceIndex = 0,
                                    Delta = new StreamDelta(),
                                    FinishReason = finishReason,
                                    Usage = ConvertStreamingUsage(initialUsage, messageDelta.Usage),
                                    CompletedMessage = completedMessage
                                }, transportResponse.Metadata, ref transportAttached);
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
                CacheWriteTokens = initial?.CacheCreationInputTokens ?? final?.CacheCreationInputTokens,
                CacheWrite5MinuteTokens = initial?.CacheCreation?.Ephemeral5mInputTokens ??
                    initial?.Ephemeral5mInputTokens ?? final?.CacheCreation?.Ephemeral5mInputTokens ?? final?.Ephemeral5mInputTokens,
                CacheWrite1HourTokens = initial?.CacheCreation?.Ephemeral1hInputTokens ??
                    initial?.Ephemeral1hInputTokens ?? final?.CacheCreation?.Ephemeral1hInputTokens ?? final?.Ephemeral1hInputTokens,
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

        private static StreamChunk AttachTransport(
            StreamChunk chunk,
            TransportMetadata metadata,
            ref bool attached)
        {
            if (!attached)
            {
                chunk.Transport = metadata;
                attached = true;
            }
            return chunk;
        }

        private static string CreateTokenCountJson(
            ClaudeMessageRequest request,
            JsonSerializerOptions options)
        {
            var json = JsonSerializer.SerializeToNode(request, options)?.AsObject()
                ?? throw new InvalidOperationException("Failed to serialize Claude token count request");

            // Anthropic's count endpoint accepts the input-bearing Messages
            // fields, including system, tools, output_config, and thinking.
            // Generation-only and bookkeeping fields are intentionally omitted.
            json.Remove("max_tokens");
            json.Remove("temperature");
            json.Remove("top_p");
            json.Remove("top_k");
            json.Remove("stop_sequences");
            json.Remove("stream");
            json.Remove("metadata");
            return json.ToJsonString(options);
        }

        private static HttpRequestMessage CreateRequest(
            string endpoint,
            string json,
            UnifiedRequest request)
        {
            var message = new HttpRequestMessage(HttpMethod.Post, endpoint)
            {
                Content = new StringContent(json, Encoding.UTF8, "application/json")
            };
            AddRequestSpecificHeaders(message, request);
            return message;
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
