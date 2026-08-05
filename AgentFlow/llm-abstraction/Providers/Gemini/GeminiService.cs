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
using LLMAbstraction.Core.Transport;
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
    public class GeminiService : ILLMServiceWithTokenCounting, IPromptCacheService
    {
        private readonly HttpClient _httpClient;
        private readonly GeminiConverter _converter;
        private readonly GeminiInteractionsConverter _interactionsConverter;
        private readonly GeminiApiMode _apiMode;
        private readonly string _apiKey;
        private readonly string _baseUrl;
        private readonly ProviderHttpTransport _transport;

        public GeminiService(
            string apiKey,
            string? baseUrl = null,
            GeminiApiMode apiMode = GeminiApiMode.Interactions,
            LLMTransportOptions? transportOptions = null)
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
            _transport = new ProviderHttpTransport(_httpClient, LLMProvider.Gemini, transportOptions);
        }

        public GeminiService(
            HttpClient httpClient, 
            string apiKey,
            GeminiConverter? converter = null,
            GeminiApiMode apiMode = GeminiApiMode.Interactions,
            GeminiInteractionsConverter? interactionsConverter = null,
            LLMTransportOptions? transportOptions = null)
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
            _transport = new ProviderHttpTransport(_httpClient, LLMProvider.Gemini, transportOptions);
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
            using var transportResponse = await _transport.SendAsync(
                () => CreateJsonRequest(endpoint, jsonContent),
                endpoint,
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
                    LLMProvider.Gemini,
                    response,
                    errorContent,
                    transportResponse.Metadata);
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
            var converted = _converter.ConvertResponse(geminiResponse);
            converted.Transport = transportResponse.Metadata;
            return converted;
        }

        public async Task<TokenCountResult> CountInputTokensAsync(
            UnifiedRequest request,
            CancellationToken cancellationToken = default)
        {
            // models.countTokens accepts generateContentRequest even when the
            // eventual generation uses Interactions. Reusing GeminiConverter
            // keeps messages, media, system instructions, tools, schemas,
            // thinking, and native continuation normalized identically.
            var countRequest = new GeminiCountTokensRequest
            {
                GenerateContentRequest = _converter.ConvertRequest(request)
            };
            var jsonOptions = JsonOptions();
            var jsonContent = JsonSerializer.Serialize(countRequest, jsonOptions);
            var endpoint = $"models/{request.Model}:countTokens";
            using var transportResponse = await _transport.SendAsync(
                () => CreateJsonRequest(endpoint, jsonContent),
                endpoint,
                request.Model,
                false,
                cancellationToken,
                request.Transport);
            var response = transportResponse.Response;

            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(
                    LLMProvider.Gemini,
                    response,
                    errorContent,
                    transportResponse.Metadata);
            }

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            var count = JsonSerializer.Deserialize<GeminiCountTokensResponse>(responseJson, jsonOptions)
                ?? throw new InvalidOperationException("Failed to deserialize Gemini token count response");
            var metadata = new Dictionary<string, object>();
            var generationUsesGenerateContent =
                _apiMode == GeminiApiMode.GenerateContent || RequiresGenerateContent(request);
            metadata["gemini.countSurface"] = "generateContent";
            metadata["gemini.generationSurface"] = generationUsesGenerateContent
                ? "generateContent"
                : "interactions";
            if (count.PromptTokensDetails != null)
                metadata["gemini.promptTokensDetails"] = count.PromptTokensDetails;
            if (count.CacheTokensDetails != null)
                metadata["gemini.cacheTokensDetails"] = count.CacheTokensDetails;

            return new TokenCountResult
            {
                Provider = LLMProvider.Gemini,
                Model = request.Model,
                InputTokens = count.TotalTokens,
                CachedTokens = count.CachedContentTokenCount,
                // Gemini exposes token counting only through models.countTokens.
                // That is the exact count request for generateContent, but an
                // Interactions request has different server framing and is
                // therefore represented honestly as a preflight estimate.
                Accuracy = generationUsesGenerateContent
                    ? TokenCountAccuracy.Exact
                    : TokenCountAccuracy.Estimate,
                ProviderMetadata = metadata,
                Transport = transportResponse.Metadata
            };
        }

        public async Task<PromptCacheResource> CreatePromptCacheAsync(
            PromptCacheCreateRequest request,
            CancellationToken cancellationToken = default)
        {
            ValidateCacheCreation(request);
            var payload = _converter.ConvertCachedContent(request);
            return await SendCacheResourceAsync(
                HttpMethod.Post,
                "cachedContents",
                payload,
                request.Prefix.Model,
                request.Transport,
                cancellationToken);
        }

        public Task<PromptCacheResource> GetPromptCacheAsync(
            string name,
            CancellationToken cancellationToken = default) =>
            SendCacheResourceAsync(
                HttpMethod.Get,
                ValidateCacheResourceName(name),
                null,
                "cached-content",
                null,
                cancellationToken);

        public async Task<PromptCacheResourcePage> ListPromptCachesAsync(
            int? pageSize = null,
            string? pageToken = null,
            CancellationToken cancellationToken = default)
        {
            if (pageSize is <= 0 or > 1000)
                throw new ArgumentOutOfRangeException(nameof(pageSize), "Gemini page size must be between 1 and 1000.");
            var query = new List<string>();
            if (pageSize != null) query.Add($"pageSize={pageSize.Value}");
            if (!string.IsNullOrEmpty(pageToken)) query.Add($"pageToken={Uri.EscapeDataString(pageToken)}");
            var endpoint = "cachedContents" + (query.Count > 0 ? "?" + string.Join("&", query) : string.Empty);
            using var transportResponse = await _transport.SendAsync(
                () => new HttpRequestMessage(HttpMethod.Get, endpoint),
                endpoint,
                "cached-content",
                false,
                cancellationToken);
            await EnsureSuccessAsync(transportResponse, cancellationToken);
            var json = await transportResponse.Response.Content.ReadAsStringAsync(cancellationToken);
            var list = JsonSerializer.Deserialize<GeminiCachedContentList>(json, JsonOptions())
                ?? throw new InvalidOperationException("Failed to deserialize Gemini cached-content list.");
            return new PromptCacheResourcePage
            {
                Items = list.CachedContents.Select(item => ConvertCacheResource(item, transportResponse.Metadata)).ToList(),
                NextPageToken = list.NextPageToken,
                Transport = transportResponse.Metadata
            };
        }

        public Task<PromptCacheResource> UpdatePromptCacheExpirationAsync(
            string name,
            PromptCacheExpiration expiration,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(expiration);
            ValidateExpiration(expiration.Ttl, expiration.ExpireTime);
            var endpoint = ValidateCacheResourceName(name);
            var updateMask = expiration.Ttl != null ? "ttl" : "expireTime";
            var payload = new GeminiCachedContent
            {
                Ttl = expiration.Ttl == null ? null : FormatDuration(expiration.Ttl.Value),
                ExpireTime = expiration.ExpireTime
            };
            return SendCacheResourceAsync(
                HttpMethod.Patch,
                $"{endpoint}?updateMask={updateMask}",
                payload,
                "cached-content",
                null,
                cancellationToken);
        }

        public async Task DeletePromptCacheAsync(
            string name,
            CancellationToken cancellationToken = default)
        {
            var endpoint = ValidateCacheResourceName(name);
            using var transportResponse = await _transport.SendAsync(
                () => new HttpRequestMessage(HttpMethod.Delete, endpoint),
                endpoint,
                "cached-content",
                false,
                cancellationToken);
            await EnsureSuccessAsync(transportResponse, cancellationToken);
        }

        private async Task<PromptCacheResource> SendCacheResourceAsync(
            HttpMethod method,
            string endpoint,
            GeminiCachedContent? payload,
            string model,
            RequestTransportOptions? transport,
            CancellationToken cancellationToken)
        {
            var json = payload == null ? null : JsonSerializer.Serialize(payload, JsonOptions());
            using var transportResponse = await _transport.SendAsync(
                () => CreateRequest(method, endpoint, json),
                endpoint,
                model,
                false,
                cancellationToken,
                transport);
            await EnsureSuccessAsync(transportResponse, cancellationToken);
            var responseJson = await transportResponse.Response.Content.ReadAsStringAsync(cancellationToken);
            var resource = JsonSerializer.Deserialize<GeminiCachedContent>(responseJson, JsonOptions())
                ?? throw new InvalidOperationException("Failed to deserialize Gemini cached-content resource.");
            return ConvertCacheResource(resource, transportResponse.Metadata);
        }

        private static PromptCacheResource ConvertCacheResource(
            GeminiCachedContent resource,
            TransportMetadata transport) => new()
        {
            Provider = LLMProvider.Gemini,
            Name = resource.Name ?? string.Empty,
            Model = resource.Model ?? string.Empty,
            DisplayName = resource.DisplayName,
            CreateTime = resource.CreateTime,
            UpdateTime = resource.UpdateTime,
            ExpireTime = resource.ExpireTime,
            CachedTokens = resource.UsageMetadata?.TotalTokenCount,
            Native = JsonSerializer.SerializeToElement(resource, JsonOptions()),
            Transport = transport
        };

        private async Task EnsureSuccessAsync(
            TransportResponse transportResponse,
            CancellationToken cancellationToken)
        {
            if (transportResponse.Response.IsSuccessStatusCode)
                return;
            var error = await transportResponse.Response.Content.ReadAsStringAsync(cancellationToken);
            throw ProviderErrorParser.Create(
                LLMProvider.Gemini,
                transportResponse.Response,
                error,
                transportResponse.Metadata);
        }

        private static void ValidateCacheCreation(PromptCacheCreateRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            ArgumentNullException.ThrowIfNull(request.Prefix);
            ValidateExpiration(request.Ttl, request.ExpireTime, allowNeither: true);
            if (request.DisplayName?.Length > 128)
                throw new ArgumentException("Gemini cached-content DisplayName cannot exceed 128 Unicode characters.", nameof(request));
            if (request.Prefix.Messages.SelectMany(message => message.Content).Any(block => block.Cache != null) ||
                request.Prefix.InstructionsCache != null ||
                request.Prefix.Tools?.Any(tool => tool.Cache != null) == true)
                throw new ArgumentException("Gemini cached-content resources do not support per-block cache directives.", nameof(request));
        }

        private static void ValidateExpiration(
            TimeSpan? ttl,
            DateTimeOffset? expireTime,
            bool allowNeither = false)
        {
            if (ttl != null && expireTime != null)
                throw new ArgumentException("Specify either Ttl or ExpireTime, not both.");
            if (!allowNeither && ttl == null && expireTime == null)
                throw new ArgumentException("Ttl or ExpireTime is required.");
            if (ttl <= TimeSpan.Zero)
                throw new ArgumentOutOfRangeException(nameof(ttl), "TTL must be greater than zero.");
        }

        private static string ValidateCacheResourceName(string name)
        {
            if (string.IsNullOrWhiteSpace(name) ||
                !name.StartsWith("cachedContents/", StringComparison.Ordinal) ||
                name.Length == "cachedContents/".Length ||
                name["cachedContents/".Length..].Contains('/') ||
                name.Contains('?') || name.Contains('#'))
                throw new ArgumentException("Cache name must have the form cachedContents/{id}.", nameof(name));
            return name;
        }

        private static string FormatDuration(TimeSpan duration)
        {
            var seconds = duration.TotalSeconds;
            return seconds == Math.Truncate(seconds)
                ? $"{seconds:0}s"
                : $"{seconds:0.#########}s";
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
            var endpoint = $"models/{request.Model}:streamGenerateContent?alt=sse";
            using var transportResponse = await _transport.SendAsync(
                () => CreateJsonRequest(endpoint, jsonContent),
                endpoint,
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
                    LLMProvider.Gemini,
                    response,
                    errorContent,
                    transportResponse.Metadata);
            }

            // Stream response as server-sent events.
            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);
            var accumulatedMessages = new Dictionary<int, UnifiedMessage>();
            var transportAttached = false;

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
                        yield return AttachTransport(new StreamChunk
                        {
                            Id = chunk.ResponseId ?? string.Empty,
                            Model = chunk.ModelVersion ?? request.Model,
                            ChoiceIndex = 0,
                            Delta = new StreamDelta(),
                            Usage = ConvertUsage(chunk.UsageMetadata)
                        }, transportResponse.Metadata, ref transportAttached);
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

                    yield return AttachTransport(
                        ConvertStreamChunk(request.Model, chunk, candidate, accumulatedMessage),
                        transportResponse.Metadata,
                        ref transportAttached);
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
                ReasoningTokens = usage.ThoughtsTokenCount,
                CacheReadTokens = usage.CachedContentTokenCount
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
            var requestJson = JsonSerializer.Serialize(nativeRequest, JsonOptions());
            using var transportResponse = await _transport.SendAsync(
                () => CreateJsonRequest("interactions", requestJson),
                "interactions",
                request.Model,
                false,
                cancellationToken,
                request.Transport);
            var response = transportResponse.Response;
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(
                    LLMProvider.Gemini,
                    response,
                    errorContent,
                    transportResponse.Metadata);
            }

            var responseJson = await response.Content.ReadAsStringAsync(cancellationToken);
            var interaction = JsonSerializer.Deserialize<GeminiInteractionResponse>(responseJson, JsonOptions());
            if (interaction == null)
                throw new InvalidOperationException("Failed to deserialize Gemini Interactions response");

            var unified = _interactionsConverter.ConvertResponse(interaction);
            unified.Transport = transportResponse.Metadata;
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
            var requestJson = JsonSerializer.Serialize(nativeRequest, JsonOptions());
            using var transportResponse = await _transport.SendAsync(
                () => CreateJsonRequest("interactions", requestJson),
                "interactions",
                request.Model,
                true,
                cancellationToken,
                request.Transport);
            var response = transportResponse.Response;
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                throw ProviderErrorParser.Create(
                    LLMProvider.Gemini,
                    response,
                    errorContent,
                    transportResponse.Metadata);
            }

            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);
            var transportAttached = false;
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
                        yield return AttachTransport(new StreamChunk
                        {
                            Model = request.Model,
                            ChoiceIndex = streamEvent.Index ?? 0,
                            Delta = new StreamDelta { ContentBlocks = projected }
                        }, transportResponse.Metadata, ref transportAttached);
                    }
                }

                if (streamEvent.EventType == "step.delta" && streamEvent.Delta is JsonElement delta)
                {
                    var type = delta.TryGetProperty("type", out var typeValue)
                        ? typeValue.GetString()
                        : null;
                    if (type == "text" && delta.TryGetProperty("text", out var textValue))
                    {
                        yield return AttachTransport(new StreamChunk
                        {
                            Model = request.Model,
                            ChoiceIndex = 0,
                            Delta = new StreamDelta { Content = textValue.GetString() }
                        }, transportResponse.Metadata, ref transportAttached);
                    }
                    else if (type == "function_call" || type == "function_call_arguments")
                    {
                        yield return AttachTransport(new StreamChunk
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
                        }, transportResponse.Metadata, ref transportAttached);
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
                    yield return AttachTransport(new StreamChunk
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
                    }, transportResponse.Metadata, ref transportAttached);
                }
            }
        }

        private static HttpRequestMessage CreateJsonRequest(string endpoint, string json) =>
            CreateRequest(HttpMethod.Post, endpoint, json);

        private static HttpRequestMessage CreateRequest(
            HttpMethod method,
            string endpoint,
            string? json)
        {
            var request = new HttpRequestMessage(method, endpoint);
            if (json != null)
                request.Content = new StringContent(json, Encoding.UTF8, "application/json");
            return request;
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
            return request.Parameters.Temperature != null ||
                request.Parameters.TopP != null ||
                request.Parameters.TopK != null ||
                request.Reasoning?.Gemini?.ThinkingBudget != null ||
                !string.IsNullOrWhiteSpace(request.Cache?.Gemini?.CachedContentName) ||
                request.Tools?.OfType<ProviderTool>().Any(tool =>
                tool.Capability == ProviderToolCapability.WebSearch &&
                tool.Options is WebSearchOptions
                {
                    Gemini: { StartTime: not null } or { EndTime: not null }
                }) == true;
        }
    }
}
