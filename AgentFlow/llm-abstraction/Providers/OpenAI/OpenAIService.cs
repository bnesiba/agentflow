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
using LLMAbstraction.Core.Errors;
using LLMAbstraction.Core;
using LLMAbstraction.Providers.OpenAI.Models;

namespace LLMAbstraction.Providers.OpenAI
{
    /// <summary>
    /// Service for interacting with the OpenAI Responses API.
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
                BaseAddress = CreateBaseAddress(_baseUrl)
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
            _httpClient.BaseAddress = CreateBaseAddress(_baseUrl);

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
            var openAIRequest = _converter.ConvertRequest(request);
            openAIRequest.Stream = false;

            var jsonOptions = CreateJsonOptions();
            var jsonContent = JsonSerializer.Serialize(openAIRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            using var response = await _httpClient.PostAsync(
                "responses",
                content,
                cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(LLMProvider.OpenAI, response, errorContent);
            }

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            var openAIResponse = JsonSerializer.Deserialize<OpenAIResponse>(
                responseJson,
                jsonOptions);

            if (openAIResponse == null)
            {
                throw new InvalidOperationException("Failed to deserialize OpenAI response");
            }

            return _converter.ConvertResponse(openAIResponse);
        }

        public async IAsyncEnumerable<StreamChunk> StreamAsync(
            UnifiedRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var openAIRequest = _converter.ConvertRequest(request);
            openAIRequest.Stream = true;

            var jsonOptions = CreateJsonOptions();
            var jsonContent = JsonSerializer.Serialize(openAIRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, "responses")
            {
                Content = content
            };
            using var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(LLMProvider.OpenAI, response, errorContent);
            }

            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);
            var streamState = new OpenAIStreamState();

            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data: "))
                    continue;

                var data = line.Substring(6);
                if (data == "[DONE]")
                    break;

                OpenAIResponseStreamEvent? streamEvent;
                try
                {
                    streamEvent = JsonSerializer.Deserialize<OpenAIResponseStreamEvent>(data, jsonOptions);
                }
                catch (JsonException exception)
                {
                    throw new InvalidOperationException(
                        "Failed to deserialize an OpenAI Responses streaming event.",
                        exception);
                }

                if (streamEvent == null)
                    continue;

                var converted = ConvertStreamEvent(streamEvent, streamState);
                if (converted != null)
                {
                    yield return converted;
                }
            }
        }

        private static JsonSerializerOptions CreateJsonOptions()
        {
            return new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
        }

        private static Uri CreateBaseAddress(string baseUrl)
        {
            return new Uri(baseUrl.EndsWith("/", StringComparison.Ordinal)
                ? baseUrl
                : baseUrl + "/");
        }

        private StreamChunk? ConvertStreamEvent(
            OpenAIResponseStreamEvent streamEvent,
            OpenAIStreamState state)
        {
            if (streamEvent.Response != null)
            {
                state.ResponseId = streamEvent.Response.Id;
                state.Model = streamEvent.Response.Model;
            }
            else if (!string.IsNullOrEmpty(streamEvent.ResponseId))
            {
                state.ResponseId = streamEvent.ResponseId;
            }

            switch (streamEvent.Type)
            {
                case "response.output_item.added":
                    if (streamEvent.Item?.Type == "web_search_call")
                    {
                        var searchMessage = new UnifiedMessage(MessageRole.Assistant, new List<ContentBlock>());
                        OpenAIConverter.AddWebSearchContent(searchMessage, streamEvent.Item);
                        return new StreamChunk
                        {
                            Id = state.ResponseId,
                            Model = state.Model,
                            ChoiceIndex = streamEvent.OutputIndex ?? 0,
                            Delta = new StreamDelta { ContentBlocks = searchMessage.Content }
                        };
                    }
                    if (streamEvent.Item?.Type != "function_call")
                        return null;

                    var outputIndex = streamEvent.OutputIndex ?? 0;
                    state.ToolCalls[outputIndex] = streamEvent.Item;
                    return new StreamChunk
                    {
                        Id = state.ResponseId,
                        Model = state.Model,
                        ChoiceIndex = outputIndex,
                        Delta = new StreamDelta
                        {
                            ToolCalls = new List<ToolCallDelta>
                            {
                                new ToolCallDelta
                                {
                                    Index = outputIndex,
                                    Id = streamEvent.Item.CallId,
                                    ItemId = streamEvent.Item.Id,
                                    Name = streamEvent.Item.Name,
                                    Type = "function",
                                    IsComplete = false
                                }
                            }
                        }
                    };

                case "response.output_text.delta":
                    return new StreamChunk
                    {
                        Id = state.ResponseId,
                        Model = state.Model,
                        ChoiceIndex = streamEvent.OutputIndex ?? 0,
                        Delta = new StreamDelta
                        {
                            Content = streamEvent.Delta
                        }
                    };

                case "response.function_call_arguments.delta":
                    state.ToolCalls.TryGetValue(streamEvent.OutputIndex ?? 0, out var activeCall);
                    return new StreamChunk
                    {
                        Id = state.ResponseId,
                        Model = state.Model,
                        ChoiceIndex = streamEvent.OutputIndex ?? 0,
                        Delta = new StreamDelta
                        {
                            ToolCalls = new List<ToolCallDelta>
                            {
                                new ToolCallDelta
                                {
                                    Index = streamEvent.OutputIndex ?? 0,
                                    Id = activeCall?.CallId,
                                    ItemId = streamEvent.ItemId ?? activeCall?.Id,
                                    Name = activeCall?.Name,
                                    Arguments = streamEvent.Delta,
                                    Type = "function"
                                }
                            }
                        }
                    };

                case "response.function_call_arguments.done":
                    state.ToolCalls.TryGetValue(streamEvent.OutputIndex ?? 0, out var completedCall);
                    state.CompletedToolCalls.Add(streamEvent.OutputIndex ?? 0);
                    return new StreamChunk
                    {
                        Id = state.ResponseId,
                        Model = state.Model,
                        ChoiceIndex = streamEvent.OutputIndex ?? 0,
                        Delta = new StreamDelta
                        {
                            ToolCalls = new List<ToolCallDelta>
                            {
                                new ToolCallDelta
                                {
                                    Index = streamEvent.OutputIndex ?? 0,
                                    Id = completedCall?.CallId,
                                    ItemId = streamEvent.ItemId ?? completedCall?.Id,
                                    Name = streamEvent.Name ?? completedCall?.Name,
                                    CompleteArguments = streamEvent.Arguments,
                                    Type = "function",
                                    IsComplete = true
                                }
                            }
                        }
                    };

                case "response.output_item.done":
                    if (streamEvent.Item?.Type != "function_call")
                        return null;

                    var completedIndex = streamEvent.OutputIndex ?? 0;
                    state.ToolCalls[completedIndex] = streamEvent.Item;
                    if (!state.CompletedToolCalls.Add(completedIndex))
                        return null;

                    return new StreamChunk
                    {
                        Id = state.ResponseId,
                        Model = state.Model,
                        ChoiceIndex = completedIndex,
                        Delta = new StreamDelta
                        {
                            ToolCalls = new List<ToolCallDelta>
                            {
                                new ToolCallDelta
                                {
                                    Index = completedIndex,
                                    Id = streamEvent.Item.CallId,
                                    ItemId = streamEvent.Item.Id,
                                    Name = streamEvent.Item.Name,
                                    CompleteArguments = streamEvent.Item.Arguments,
                                    Type = "function",
                                    IsComplete = true
                                }
                            }
                        }
                    };

                case "response.completed":
                case "response.incomplete":
                    return new StreamChunk
                    {
                        Id = state.ResponseId,
                        Model = state.Model,
                        ChoiceIndex = 0,
                        Delta = new StreamDelta(),
                        FinishReason = streamEvent.Response != null
                            ? ConvertFinishReason(streamEvent.Response)
                            : FinishReason.Stop,
                        Usage = streamEvent.Response?.Usage != null ? ConvertUsage(streamEvent.Response.Usage) : null,
                        Continuation = streamEvent.Response != null
                            ? OpenAIConverter.CreateContinuationState(streamEvent.Response)
                            : null,
                        CompletedMessage = streamEvent.Response != null
                            ? _converter.ConvertResponse(streamEvent.Response).Choices[0].Message
                            : null
                    };

                case "response.failed":
                    return new StreamChunk
                    {
                        Id = state.ResponseId,
                        Model = state.Model,
                        ChoiceIndex = 0,
                        Delta = new StreamDelta(),
                        FinishReason = FinishReason.Error,
                        Usage = streamEvent.Response?.Usage != null ? ConvertUsage(streamEvent.Response.Usage) : null,
                        Continuation = streamEvent.Response != null
                            ? OpenAIConverter.CreateContinuationState(streamEvent.Response)
                            : null
                    };

                default:
                    return null;
            }
        }

        private sealed class OpenAIStreamState
        {
            public string ResponseId { get; set; } = string.Empty;
            public string Model { get; set; } = string.Empty;
            public Dictionary<int, OpenAIOutputItem> ToolCalls { get; } = new();
            public HashSet<int> CompletedToolCalls { get; } = new();
        }

        private static UsageInfo ConvertUsage(OpenAIResponseUsage usage)
        {
            return new UsageInfo
            {
                InputTokens = usage.InputTokens,
                OutputTokens = usage.OutputTokens,
                TotalTokens = usage.TotalTokens,
                CacheReadTokens = usage.InputTokenDetails?.CachedTokens,
                ReasoningTokens = usage.OutputTokenDetails?.ReasoningTokens
            };
        }

        private static FinishReason ConvertFinishReason(OpenAIResponse response)
        {
            if (response.Output.Exists(item => item.Type == "function_call"))
                return FinishReason.ToolCalls;

            if (response.Status == "incomplete")
            {
                return response.IncompleteDetails?.Reason switch
                {
                    "max_output_tokens" => FinishReason.MaxTokens,
                    "content_filter" => FinishReason.ContentFilter,
                    _ => FinishReason.Other
                };
            }

            if (response.Error != null)
                return FinishReason.Error;

            return FinishReason.Stop;
        }
    }
}
