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

            var jsonOptions = CreateJsonOptions();
            var jsonContent = JsonSerializer.Serialize(openAIRequest, jsonOptions);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync(
                "responses",
                content,
                cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException(
                    $"OpenAI API request failed with status {response.StatusCode}: {errorContent}");
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
            var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw new HttpRequestException(
                    $"OpenAI API request failed with status {response.StatusCode}: {errorContent}");
            }

            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);

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
                catch
                {
                    continue;
                }

                if (streamEvent == null)
                    continue;

                var converted = ConvertStreamEvent(streamEvent);
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

        private static StreamChunk? ConvertStreamEvent(OpenAIResponseStreamEvent streamEvent)
        {
            switch (streamEvent.Type)
            {
                case "response.output_text.delta":
                    return new StreamChunk
                    {
                        Id = streamEvent.ResponseId ?? streamEvent.Response?.Id ?? string.Empty,
                        Model = streamEvent.Response?.Model ?? string.Empty,
                        ChoiceIndex = streamEvent.OutputIndex ?? 0,
                        Delta = new StreamDelta
                        {
                            Content = streamEvent.Delta
                        }
                    };

                case "response.function_call_arguments.delta":
                    return new StreamChunk
                    {
                        Id = streamEvent.ResponseId ?? string.Empty,
                        ChoiceIndex = streamEvent.OutputIndex ?? 0,
                        Delta = new StreamDelta
                        {
                            ToolCalls = new List<ToolCallDelta>
                            {
                                new ToolCallDelta
                                {
                                    Index = streamEvent.OutputIndex ?? 0,
                                    Id = streamEvent.ItemId,
                                    Arguments = streamEvent.Delta,
                                    Type = "function"
                                }
                            }
                        }
                    };

                case "response.function_call_arguments.done":
                    return new StreamChunk
                    {
                        Id = streamEvent.ResponseId ?? string.Empty,
                        ChoiceIndex = streamEvent.OutputIndex ?? 0,
                        Delta = new StreamDelta
                        {
                            ToolCalls = new List<ToolCallDelta>
                            {
                                new ToolCallDelta
                                {
                                    Index = streamEvent.OutputIndex ?? 0,
                                    Id = streamEvent.ItemId,
                                    Name = streamEvent.Name,
                                    Arguments = streamEvent.Arguments,
                                    Type = "function"
                                }
                            }
                        }
                    };

                case "response.completed":
                case "response.incomplete":
                    return new StreamChunk
                    {
                        Id = streamEvent.Response?.Id ?? streamEvent.ResponseId ?? string.Empty,
                        Model = streamEvent.Response?.Model ?? string.Empty,
                        ChoiceIndex = 0,
                        Delta = new StreamDelta(),
                        FinishReason = streamEvent.Response != null
                            ? ConvertFinishReason(streamEvent.Response)
                            : FinishReason.Stop,
                        Usage = streamEvent.Response?.Usage != null ? ConvertUsage(streamEvent.Response.Usage) : null
                    };

                case "response.failed":
                    return new StreamChunk
                    {
                        Id = streamEvent.Response?.Id ?? streamEvent.ResponseId ?? string.Empty,
                        Model = streamEvent.Response?.Model ?? string.Empty,
                        ChoiceIndex = 0,
                        Delta = new StreamDelta(),
                        FinishReason = FinishReason.Error,
                        Usage = streamEvent.Response?.Usage != null ? ConvertUsage(streamEvent.Response.Usage) : null
                    };

                default:
                    return null;
            }
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
