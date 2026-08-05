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
    public enum GeminiApiMode
    {
        Interactions,
        GenerateContent
    }

    /// <summary>
    /// Service for interacting with Gemini API
    /// </summary>
    public class GeminiService : ILLMService
    {
        private readonly HttpClient _httpClient;
        private readonly GeminiConverter _converter;
        private readonly GeminiInteractionsConverter _interactionsConverter;
        private readonly GeminiApiMode _apiMode;
        private readonly string _apiKey;
        private readonly string _baseUrl;

        public GeminiService(
            string apiKey,
            string? baseUrl = null,
            GeminiApiMode apiMode = GeminiApiMode.Interactions)
        {
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _baseUrl = baseUrl ?? "https://generativelanguage.googleapis.com/v1beta";
            _converter = new GeminiConverter();
            _interactionsConverter = new GeminiInteractionsConverter();
            _apiMode = apiMode;
            
            _httpClient = new HttpClient
            {
                BaseAddress = CreateBaseAddress(_baseUrl)
            };
            ConfigureHeaders();
        }

        public GeminiService(
            HttpClient httpClient, 
            string apiKey,
            GeminiConverter? converter = null,
            GeminiApiMode apiMode = GeminiApiMode.Interactions,
            GeminiInteractionsConverter? interactionsConverter = null)
        {
            _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
            _baseUrl = httpClient.BaseAddress?.ToString() ?? 
                "https://generativelanguage.googleapis.com/v1beta";
            _converter = converter ?? new GeminiConverter();
            _interactionsConverter = interactionsConverter ?? new GeminiInteractionsConverter();
            _apiMode = apiMode;
            _httpClient.BaseAddress = CreateBaseAddress(_baseUrl);
            ConfigureHeaders();
        }

        public async Task<UnifiedResponse> GenerateAsync(
            UnifiedRequest request, 
            CancellationToken cancellationToken = default)
        {
            if (_apiMode == GeminiApiMode.Interactions && !RequiresGenerateContent(request))
                return await GenerateInteractionAsync(request, cancellationToken);

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
            if (_apiMode == GeminiApiMode.Interactions && !RequiresGenerateContent(request))
            {
                await foreach (var chunk in StreamInteractionAsync(request, cancellationToken))
                    yield return chunk;
                yield break;
            }

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

            var contentCountBeforeGrounding = accumulatedMessage.Content.Count;
            GeminiConverter.ProjectGroundingMetadata(
                candidate.GroundingMetadata,
                accumulatedMessage,
                $"google_search_{candidate.Index}");
            foreach (var groundingBlock in accumulatedMessage.Content.Skip(contentCountBeforeGrounding))
                delta.ContentBlocks.Add(groundingBlock);

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

        private async Task<UnifiedResponse> GenerateInteractionAsync(
            UnifiedRequest request,
            CancellationToken cancellationToken)
        {
            var nativeRequest = _interactionsConverter.ConvertRequest(request);
            var content = SerializeContent(nativeRequest);
            using var response = await _httpClient.PostAsync("interactions", content, cancellationToken);
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(LLMProvider.Gemini, response, errorContent);
            }

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            var interaction = JsonSerializer.Deserialize<GeminiInteractionResponse>(responseJson, JsonOptions());
            if (interaction == null)
                throw new InvalidOperationException("Failed to deserialize Gemini Interactions response");

            var unified = _interactionsConverter.ConvertResponse(interaction);
            if (nativeRequest.Store && unified.Continuation != null)
            {
                unified.Continuation.Mode = ContinuationMode.ServerManaged;
                unified.Continuation.NativeItems.Clear();
            }
            return unified;
        }

        private async IAsyncEnumerable<StreamChunk> StreamInteractionAsync(
            UnifiedRequest request,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var nativeRequest = _interactionsConverter.ConvertRequest(request);
            nativeRequest.Stream = true;
            using var httpRequest = new HttpRequestMessage(HttpMethod.Post, "interactions")
            {
                Content = SerializeContent(nativeRequest)
            };
            using var response = await _httpClient.SendAsync(
                httpRequest,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(LLMProvider.Gemini, response, errorContent);
            }

            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);
            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync(cancellationToken);
                if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("data: ", StringComparison.Ordinal))
                    continue;

                GeminiInteractionStreamEvent? streamEvent;
                try
                {
                    streamEvent = JsonSerializer.Deserialize<GeminiInteractionStreamEvent>(
                        line.Substring(6),
                        JsonOptions());
                }
                catch (JsonException exception)
                {
                    throw new InvalidOperationException(
                        "Failed to deserialize a Gemini Interactions streaming response event.",
                        exception);
                }

                if (streamEvent == null)
                    continue;

                if (streamEvent.EventType == "error")
                    throw new InvalidOperationException($"Gemini Interactions stream failed: {streamEvent.Error}");

                if (streamEvent.EventType is "step.start" or "step.stop" &&
                    streamEvent.Step is JsonElement step)
                {
                    var stepType = GetString(step, "type");
                    var projected = new List<ContentBlock>();
                    GeminiInteractionsConverter.ProjectStep(step, projected);
                    var shouldEmit =
                        (streamEvent.EventType == "step.start" && stepType == "google_search_call") ||
                        (streamEvent.EventType == "step.stop" && stepType == "google_search_result");
                    if (shouldEmit)
                    {
                        yield return new StreamChunk
                        {
                            Model = request.Model,
                            ChoiceIndex = streamEvent.Index ?? 0,
                            Delta = new StreamDelta { ContentBlocks = projected }
                        };
                    }
                }

                if (streamEvent.EventType == "step.delta" && streamEvent.Delta is JsonElement delta)
                {
                    var type = delta.TryGetProperty("type", out var typeValue)
                        ? typeValue.GetString()
                        : null;
                    if (type == "text" && delta.TryGetProperty("text", out var textValue))
                    {
                        yield return new StreamChunk
                        {
                            Model = request.Model,
                            ChoiceIndex = 0,
                            Delta = new StreamDelta { Content = textValue.GetString() }
                        };
                    }
                    else if (type == "function_call" || type == "function_call_arguments")
                    {
                        yield return new StreamChunk
                        {
                            Model = request.Model,
                            ChoiceIndex = 0,
                            Delta = new StreamDelta
                            {
                                ToolCalls = new List<ToolCallDelta>
                                {
                                    new()
                                    {
                                        Index = streamEvent.Index ?? 0,
                                        Id = GetString(delta, "id"),
                                        Name = GetString(delta, "name"),
                                        Arguments = delta.TryGetProperty("arguments", out var arguments)
                                            ? arguments.ValueKind == JsonValueKind.String
                                                ? arguments.GetString()
                                                : arguments.GetRawText()
                                            : null,
                                        Type = "function"
                                    }
                                }
                            }
                        };
                    }
                }

                if (streamEvent.EventType == "interaction.completed" && streamEvent.Interaction != null)
                {
                    var completed = _interactionsConverter.ConvertResponse(streamEvent.Interaction);
                    if (nativeRequest.Store && completed.Continuation != null)
                    {
                        completed.Continuation.Mode = ContinuationMode.ServerManaged;
                        completed.Continuation.NativeItems.Clear();
                    }
                    var choice = completed.Choices.FirstOrDefault();
                    yield return new StreamChunk
                    {
                        Id = completed.Id,
                        Model = completed.Model,
                        ChoiceIndex = 0,
                        Delta = new StreamDelta(),
                        FinishReason = choice?.FinishReason ?? FinishReason.Stop,
                        Usage = completed.Usage,
                        CompletedMessage = choice?.Message,
                        Continuation = completed.Continuation,
                        ProviderMetadata = completed.ProviderMetadata
                    };
                }
            }
        }

        private static StringContent SerializeContent(object value) => new(
            JsonSerializer.Serialize(value, JsonOptions()),
            Encoding.UTF8,
            "application/json");

        private static JsonSerializerOptions JsonOptions() => new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };

        private static string? GetString(JsonElement element, string property) =>
            element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.String
                ? value.GetString()
                : null;

        private static bool RequiresGenerateContent(UnifiedRequest request)
        {
            return request.Tools?.OfType<ProviderTool>().Any(tool =>
                tool.Capability == ProviderToolCapability.WebSearch &&
                tool.Options is WebSearchOptions
                {
                    Gemini: { StartTime: not null } or { EndTime: not null }
                }) == true;
        }
    }
}
