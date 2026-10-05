using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Core;
using LLMAbstraction.Providers.Gemini.Models;

namespace LLMAbstraction.Providers.Gemini
{
    /// <summary>
    /// Converts between unified models and Gemini-specific models
    /// </summary>
    public class GeminiConverter : IModelConverter<GeminiGenerateRequest, GeminiGenerateResponse>
    {
        private static readonly HashSet<string> ProtectedRequestFields = new(StringComparer.Ordinal)
        {
            "contents", "systemInstruction", "generationConfig", "tools", "toolConfig", "safetySettings",
            "cachedContent"
        };
        private static readonly HashSet<string> SeparatelyHandledOptionFields = new(StringComparer.Ordinal)
        {
            "safetySettings"
        };
        private static readonly JsonSerializerOptions NativeJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        public GeminiGenerateRequest ConvertRequest(UnifiedRequest request)
        {
            LLMRequestValidator.ValidateAndThrow(
                request,
                LLMProvider.Gemini,
                LLMApiSurface.GeminiGenerateContent);
            var geminiRequest = new GeminiGenerateRequest
            {
                Contents = ConvertMessages(request.Messages),
                GenerationConfig = new GeminiGenerationConfig
                {
                    MaxOutputTokens = request.Parameters.MaxOutputTokens,
                    Temperature = request.Parameters.Temperature,
                    TopP = request.Parameters.TopP,
                    TopK = request.Parameters.TopK,
                    StopSequences = request.Parameters.StopSequences,
                    ThinkingConfig = ConvertReasoning(request.Reasoning)
                },
                CachedContent = request.Cache?.Gemini?.CachedContentName
            };

            // Add normalized instruction text, including system-role messages.
            var combinedInstructions = UnifiedRequestNormalization.CombineInstructions(request);
            if (!string.IsNullOrEmpty(combinedInstructions))
            {
                geminiRequest.SystemInstruction = new GeminiContent
                {
                    Parts = new List<GeminiPart>
                    {
                        new GeminiPart { Text = combinedInstructions }
                    }
                };
            }

            // Convert tools if present
            var convertedTools = request.Tools?.Select(ConvertTool).ToList();
            if (convertedTools?.Count > 0)
            {
                geminiRequest.Tools = convertedTools;

                geminiRequest.ToolConfig = ConvertToolChoice(request.ToolChoice);
            }

            // Convert output format if present
            if (request.Output != null)
            {
                ConvertOutputFormat(geminiRequest, request.Output);
            }

            if (request.ProviderOptions?.Gemini != null &&
                request.ProviderOptions.Gemini.TryGetValue("safetySettings", out var safetySettings) &&
                safetySettings is List<Dictionary<string, object>> typedSafetySettings)
            {
                geminiRequest.SafetySettings = typedSafetySettings;
            }

            geminiRequest.AdditionalProperties = ProviderOptionMerger.ConvertAdditionalFields(
                request.ProviderOptions?.Gemini,
                ProtectedRequestFields,
                SeparatelyHandledOptionFields);

            return geminiRequest;
        }

        internal GeminiCachedContent ConvertCachedContent(PromptCacheCreateRequest request)
        {
            ArgumentNullException.ThrowIfNull(request);
            ArgumentNullException.ThrowIfNull(request.Prefix);
            if (string.IsNullOrWhiteSpace(request.Prefix.Model))
                throw new ArgumentException("A provider model identifier is required.", nameof(request));

            var converted = new GeminiCachedContent
            {
                Model = request.Prefix.Model.StartsWith("models/", StringComparison.Ordinal)
                    ? request.Prefix.Model
                    : $"models/{request.Prefix.Model}",
                DisplayName = request.DisplayName,
                Contents = ConvertMessages(request.Prefix.Messages),
                Ttl = request.Ttl == null ? null : FormatDuration(request.Ttl.Value),
                ExpireTime = request.ExpireTime
            };
            var instructions = UnifiedRequestNormalization.CombineInstructions(request.Prefix);
            if (!string.IsNullOrEmpty(instructions))
            {
                converted.SystemInstruction = new GeminiContent
                {
                    Parts = new List<GeminiPart> { new() { Text = instructions } }
                };
            }
            var tools = request.Prefix.Tools?.Select(ConvertTool).ToList();
            if (tools?.Count > 0)
            {
                converted.Tools = tools;
                converted.ToolConfig = ConvertToolChoice(request.Prefix.ToolChoice);
            }
            return converted;
        }

        private static string FormatDuration(TimeSpan duration)
        {
            var seconds = duration.TotalSeconds;
            return seconds == Math.Truncate(seconds)
                ? $"{seconds:0}s"
                : $"{seconds:0.#########}s";
        }

        public UnifiedResponse ConvertResponse(GeminiGenerateResponse response)
        {
            var choices = new List<ResponseChoice>();

            if (response.Candidates != null)
            {
                for (int i = 0; i < response.Candidates.Count; i++)
                {
                    var candidate = response.Candidates[i];
                    var message = ConvertContent(candidate.Content);
                    ProjectGroundingMetadata(candidate.GroundingMetadata, message, $"google_search_{candidate.Index}");
                    EvidenceProjector.Project(message, ProviderIds.Gemini);
                    choices.Add(new ResponseChoice
                    {
                        Index = candidate.Index,
                        Message = message,
                        FinishReason = message.IsAssistantWithToolCalls()
                            ? FinishReason.ToolCalls
                            : ConvertFinishReason(candidate.FinishReason),
                        ProviderMetadata = new Dictionary<string, object>
                        {
                            ["gemini.finishReason"] = candidate.FinishReason ?? string.Empty,
                            ["gemini.safetyRatings"] = candidate.SafetyRatings ?? new List<GeminiSafetyRating>()
                        }
                    });
                }
            }

            return new UnifiedResponse
            {
                Id = response.ResponseId ?? Guid.NewGuid().ToString(),
                Model = response.ModelVersion ?? string.Empty,
                Choices = choices,
                Usage = new UsageInfo
                {
                    InputTokens = response.UsageMetadata?.PromptTokenCount ?? 0,
                    OutputTokens = response.UsageMetadata?.CandidatesTokenCount ?? 0,
                    TotalTokens = response.UsageMetadata?.TotalTokenCount ?? 0,
                    ReasoningTokens = response.UsageMetadata?.ThoughtsTokenCount,
                    CacheReadTokens = response.UsageMetadata?.CachedContentTokenCount
                },
                ProviderMetadata = response.PromptFeedback != null
                    ? new Dictionary<string, object> { { "promptFeedback", response.PromptFeedback } }
                    : null
            };
        }

        private List<GeminiContent> ConvertMessages(List<UnifiedMessage> messages)
        {
            var result = new List<GeminiContent>();
            var toolNamesByCallId = new Dictionary<string, string>(StringComparer.Ordinal);

            foreach (var message in messages)
            {
                // Skip system messages (handled separately in Gemini)
                if (message.Role == MessageRole.System)
                    continue;

                var role = message.Role switch
                {
                    MessageRole.User => "user",
                    MessageRole.Assistant => "model",
                    MessageRole.Tool => "user", // Tool results go in user messages
                    _ => throw new ArgumentException($"Unknown role: {message.Role}")
                };

                var parts = ConvertContentBlocks(message.Content, toolNamesByCallId);

                result.Add(new GeminiContent
                {
                    Role = role,
                    Parts = parts
                });
            }

            return result;
        }

        private List<GeminiPart> ConvertContentBlocks(
            List<ContentBlock> content,
            Dictionary<string, string> toolNamesByCallId)
        {
            var parts = new List<GeminiPart>();

            foreach (var block in content)
            {
                if (block is ToolCallContent knownToolCall && !string.IsNullOrEmpty(knownToolCall.Id))
                    toolNamesByCallId[knownToolCall.Id] = knownToolCall.Name;

                if (TryGetGeminiNativeValue(block, out var nativeValue))
                {
                    parts.Add(JsonSerializer.Deserialize<GeminiPart>(
                        nativeValue.GetRawText(),
                        NativeJsonOptions)!);
                    continue;
                }

                switch (block)
                {
                    case TextContent text:
                        parts.Add(new GeminiPart { Text = text.Text });
                        break;

                    case ImageContent image:
                        if (!string.IsNullOrEmpty(image.Source.Data))
                        {
                            parts.Add(new GeminiPart
                            {
                                InlineData = new GeminiInlineData
                                {
                                    MimeType = image.Source.MediaType ?? "image/jpeg",
                                    Data = image.Source.Data
                                }
                            });
                        }
                        else if (!string.IsNullOrEmpty(image.Source.Url))
                        {
                            throw new NotSupportedException(
                                "Gemini ImageContent does not accept arbitrary URLs. Upload the image or use MediaContent.Source.FileUri.");
                        }
                        else
                        {
                            throw new NotSupportedException("Gemini image content requires base64 data or a Gemini file URI.");
                        }
                        break;

                    case MediaContent media:
                        AddMediaPart(parts, media);
                        break;

                    case ToolCallContent toolCall:
                        var toolCallPart = new GeminiPart
                        {
                            FunctionCall = new GeminiFunctionCall
                            {
                                Id = toolCall.Id,
                                Name = toolCall.Name,
                                Args = toolCall.Input
                            }
                        };
                        ApplyGeminiProviderMetadata(toolCallPart, toolCall.ProviderMetadata);
                        parts.Add(toolCallPart);
                        break;

                    case ToolResultContent toolResult:
                        var toolName = ResolveToolResultName(toolResult, toolNamesByCallId);
                        parts.Add(new GeminiPart
                        {
                            FunctionResponse = new GeminiFunctionResponse
                            {
                                Name = toolName,
                                Id = toolResult.ToolCallId,
                                Response = ConvertToolResultOutput(toolResult.Output)
                            }
                        });
                        break;

                    case ProviderNativeContent:
                        throw new NotSupportedException(
                            "Provider-native content cannot be sent to Gemini unless it originated from Gemini.");

                    default:
                        throw new NotSupportedException(
                            $"Content type '{block.Type}' is not supported by the Gemini converter.");
                }
            }

            return parts;
        }

        private static string ResolveToolResultName(
            ToolResultContent toolResult,
            Dictionary<string, string> toolNamesByCallId)
        {
            toolNamesByCallId.TryGetValue(toolResult.ToolCallId, out var knownName);

            if (!string.IsNullOrEmpty(toolResult.ToolName))
            {
                if (knownName != null && !string.Equals(
                    knownName,
                    toolResult.ToolName,
                    StringComparison.Ordinal))
                {
                    throw new ArgumentException(
                        $"Gemini tool result name '{toolResult.ToolName}' does not match prior call '{toolResult.ToolCallId}' name '{knownName}'.");
                }

                return toolResult.ToolName;
            }

            if (knownName != null)
                return knownName;

            throw new ArgumentException(
                $"Gemini tool result '{toolResult.ToolCallId}' requires ToolName because no matching prior function call is present in the request history.");
        }

        private GeminiTool ConvertTool(LLMTool tool)
        {
            return tool switch
            {
                FunctionTool function => new GeminiTool
                {
                    FunctionDeclarations = new List<GeminiFunctionDeclaration>
                    {
                        ConvertFunctionTool(function)
                    }
                },
                ProviderTool { Capability: ProviderToolCapability.WebSearch } provider
                    => ConvertWebSearchTool((WebSearchOptions)provider.Options),
                ProviderTool provider => throw new NotSupportedException(
                    $"Gemini provider tool '{provider.Capability}' is not implemented."),
                _ => throw new NotSupportedException($"Unknown tool type '{tool.GetType().Name}'.")
            };
        }

        private GeminiFunctionDeclaration ConvertFunctionTool(FunctionTool tool)
        {
            return new GeminiFunctionDeclaration
            {
                Name = tool.Name,
                Description = tool.Description,
                Parameters = tool.Parameters
            };
        }

        private static GeminiTool ConvertWebSearchTool(WebSearchOptions options)
        {
            var native = options.Gemini;
            GeminiTimeRangeFilter? timeRange = null;
            if (native?.StartTime != null && native.EndTime != null)
            {
                timeRange = new GeminiTimeRangeFilter
                {
                    StartTime = native.StartTime.Value.ToUniversalTime().ToString("O"),
                    EndTime = native.EndTime.Value.ToUniversalTime().ToString("O")
                };
            }

            return new GeminiTool
            {
                GoogleSearch = new GeminiGoogleSearch
                {
                    TimeRangeFilter = timeRange,
                    SearchTypes = new GeminiSearchTypes
                    {
                        WebSearch = options.ContentTypes.HasFlag(WebSearchContentTypes.Web)
                            ? new Dictionary<string, object>()
                            : null,
                        ImageSearch = options.ContentTypes.HasFlag(WebSearchContentTypes.Image)
                            ? new Dictionary<string, object>()
                            : null
                    }
                }
            };
        }

        private GeminiToolConfig? ConvertToolChoice(ToolChoice? toolChoice)
        {
            if (toolChoice == null)
                return null;

            var mode = toolChoice.Type switch
            {
                ToolChoiceType.Auto => "AUTO",
                ToolChoiceType.None => "NONE",
                ToolChoiceType.Required => "ANY",
                ToolChoiceType.Specific => "ANY",
                _ => "AUTO"
            };

            var config = new GeminiToolConfig
            {
                FunctionCallingConfig = new GeminiFunctionCallingConfig
                {
                    Mode = mode
                }
            };

            if (toolChoice.Type == ToolChoiceType.Specific && !string.IsNullOrEmpty(toolChoice.ToolName))
            {
                config.FunctionCallingConfig.AllowedFunctionNames = new List<string>
                {
                    toolChoice.ToolName
                };
            }

            return config;
        }

        private UnifiedMessage ConvertContent(GeminiContent content)
        {
            var message = new UnifiedMessage
            {
                Role = MessageRole.Assistant,
                Content = new List<ContentBlock>()
            };

            if (content.Parts != null)
            {
                foreach (var part in content.Parts)
                {
                    message.Content.Add(ConvertResponsePart(part));
                }
            }

            return message;
        }

        internal static ContentBlock ConvertResponsePart(GeminiPart part)
        {
            var nativeRepresentation = ProviderNativeRepresentation.Create(
                ProviderIds.Gemini,
                JsonSerializer.SerializeToElement(part, NativeJsonOptions));

            if (IsThoughtPart(part))
            {
                return new ProviderNativeContent
                {
                    NativeType = "thought",
                    NativeRepresentation = nativeRepresentation
                };
            }

            if (part.Text != null)
            {
                return new TextContent
                {
                    Text = part.Text,
                    NativeRepresentation = nativeRepresentation
                };
            }

            if (part.FunctionCall != null)
            {
                var toolCall = new ToolCallContent
                {
                    Id = part.FunctionCall.Id ?? Guid.NewGuid().ToString(),
                    Name = part.FunctionCall.Name,
                    Input = part.FunctionCall.Args ?? new Dictionary<string, object>(),
                    NativeRepresentation = nativeRepresentation
                };
                if (!string.IsNullOrEmpty(part.ThoughtSignature))
                {
                    toolCall.ProviderMetadata = new Dictionary<string, object>
                    {
                        { "gemini.thoughtSignature", part.ThoughtSignature }
                    };
                }

                return toolCall;
            }

            return new ProviderNativeContent
            {
                NativeType = GetNativePartType(part),
                NativeRepresentation = nativeRepresentation
            };
        }

        internal static void ProjectGroundingMetadata(
            JsonElement? groundingMetadata,
            UnifiedMessage message,
            string toolCallId)
        {
            if (groundingMetadata is not { ValueKind: JsonValueKind.Object } metadata)
                return;

            var native = ProviderNativeRepresentation.Create(ProviderIds.Gemini, metadata);
            var queries = metadata.TryGetProperty("webSearchQueries", out var queryValues) &&
                queryValues.ValueKind == JsonValueKind.Array
                    ? queryValues.EnumerateArray()
                        .Where(value => value.ValueKind == JsonValueKind.String)
                        .Select(value => value.GetString())
                        .Where(value => value != null)
                        .ToList()
                    : new List<string?>();

            message.Content.Add(new ProviderToolCallContent
            {
                Id = toolCallId,
                ToolId = "web_search",
                Capability = ProviderToolCapability.WebSearch,
                Status = "completed",
                Input = new Dictionary<string, object?> { ["queries"] = queries },
                NativeRepresentation = native
            });

            var sources = ExtractGroundingSources(metadata);
            var suggestions = metadata.TryGetProperty("searchEntryPoint", out var entryPoint) &&
                entryPoint.ValueKind == JsonValueKind.Object &&
                entryPoint.TryGetProperty("renderedContent", out var renderedContent) &&
                renderedContent.ValueKind == JsonValueKind.String
                    ? renderedContent.GetString()
                    : null;
            message.Content.Add(new ProviderToolResultContent
            {
                ToolCallId = toolCallId,
                Capability = ProviderToolCapability.WebSearch,
                Status = "completed",
                Output = JsonSerializer.Deserialize<object>(metadata.GetRawText()),
                Sources = sources,
                SearchSuggestionsHtml = suggestions,
                NativeRepresentation = native
            });

            // Assign stable source IDs before linking supports/citations.
            EvidenceProjector.Project(message, ProviderIds.Gemini);

            var text = message.Content.OfType<TextContent>().FirstOrDefault();
            if (text == null || !metadata.TryGetProperty("groundingSupports", out var supports) ||
                supports.ValueKind != JsonValueKind.Array)
                return;
            foreach (var support in supports.EnumerateArray())
            {
                if (!support.TryGetProperty("segment", out var segment) ||
                    !support.TryGetProperty("groundingChunkIndices", out var indices) ||
                    indices.ValueKind != JsonValueKind.Array)
                    continue;
                var textBlockIndex = message.Content.IndexOf(text);
                var grounding = new GroundingSupport
                {
                    AnswerSpan = CreateAnswerSpan(segment, textBlockIndex),
                    SourceConfidences = ReadDoubles(support, "confidenceScores"),
                    NativeRepresentation = ProviderNativeRepresentation.Create(ProviderIds.Gemini, support)
                };
                foreach (var index in indices.EnumerateArray())
                {
                    if (!index.TryGetInt32(out var sourceIndex) || sourceIndex < 0 || sourceIndex >= sources.Count)
                        continue;
                    var sourceId = sources[sourceIndex].SourceId;
                    if (sourceId == null)
                        continue;
                    grounding.Sources.Add(new CitationSourceReference { SourceId = sourceId });
                    text.Citations.Add(new Citation
                    {
                        Url = sources[sourceIndex].Url,
                        Title = sources[sourceIndex].Title,
                        StartIndex = GetInt(segment, "startIndex"),
                        EndIndex = GetInt(segment, "endIndex"),
                        AnswerSpan = CreateAnswerSpan(segment, textBlockIndex),
                        Sources = new List<CitationSourceReference>
                        {
                            new() { SourceId = sourceId }
                        },
                        NativeRepresentation = ProviderNativeRepresentation.Create(ProviderIds.Gemini, support)
                    });
                }
                if (grounding.Sources.Count > 0)
                    message.Evidence.GroundingSupports.Add(grounding);
            }
        }

        private static AnswerTextSpan? CreateAnswerSpan(JsonElement segment, int blockIndex)
        {
            var start = GetInt(segment, "startIndex");
            var end = GetInt(segment, "endIndex");
            return start != null && end != null
                ? new AnswerTextSpan
                {
                    ContentBlockIndex = blockIndex,
                    StartIndex = start.Value,
                    EndIndex = end.Value,
                    Unit = TextIndexUnit.ProviderDefined
                }
                : null;
        }

        private static IReadOnlyList<double>? ReadDoubles(JsonElement element, string property)
        {
            if (!element.TryGetProperty(property, out var values) || values.ValueKind != JsonValueKind.Array)
                return null;
            var result = new List<double>();
            foreach (var value in values.EnumerateArray())
                if (value.TryGetDouble(out var number))
                    result.Add(number);
            return result.Count == 0 ? null : result;
        }

        private static List<WebSource> ExtractGroundingSources(JsonElement metadata)
        {
            var result = new List<WebSource>();
            if (!metadata.TryGetProperty("groundingChunks", out var chunks) || chunks.ValueKind != JsonValueKind.Array)
                return result;
            foreach (var chunk in chunks.EnumerateArray())
            {
                if (chunk.TryGetProperty("web", out var web))
                {
                    result.Add(new WebSource
                    {
                        Url = GetString(web, "uri"),
                        Title = GetString(web, "title"),
                        SourceType = "web"
                    });
                }
                else if (chunk.TryGetProperty("image", out var image))
                {
                    result.Add(new WebSource
                    {
                        Url = GetString(image, "sourceUri") ?? GetString(image, "uri"),
                        Title = GetString(image, "title"),
                        SourceType = "image"
                    });
                }
            }
            return result;
        }

        private static string? GetString(JsonElement element, string property) =>
            element.ValueKind == JsonValueKind.Object && element.TryGetProperty(property, out var value) &&
            value.ValueKind == JsonValueKind.String ? value.GetString() : null;

        private static int? GetInt(JsonElement element, string property) =>
            element.ValueKind == JsonValueKind.Object && element.TryGetProperty(property, out var value) &&
            value.TryGetInt32(out var result) ? result : null;

        private static bool TryGetGeminiNativeValue(ContentBlock block, out JsonElement value)
        {
            if (block.NativeRepresentation?.Provider == ProviderIds.Gemini)
            {
                value = block.NativeRepresentation.Value.Clone();
                return true;
            }

            value = default;
            return false;
        }

        private static bool IsThoughtPart(GeminiPart part)
        {
            return part.AdditionalProperties != null &&
                part.AdditionalProperties.TryGetValue("thought", out var thought) &&
                thought.ValueKind == JsonValueKind.True;
        }

        private static string GetNativePartType(GeminiPart part)
        {
            if (part.InlineData != null)
                return "inlineData";
            if (part.FileData != null)
                return "fileData";
            if (part.FunctionResponse != null)
                return "functionResponse";

            return part.AdditionalProperties?.Keys.FirstOrDefault() ?? "unknown";
        }

        private static void ConvertOutputFormat(GeminiGenerateRequest request, OutputFormat output)
        {
            // Gemini uses responseMimeType and responseSchema in GenerationConfig
            if (request.GenerationConfig == null)
                request.GenerationConfig = new GeminiGenerationConfig();

            switch (output.Kind)
            {
                case OutputFormatKind.JsonObject:
                    request.GenerationConfig.ResponseMimeType = "application/json";
                    break;

                case OutputFormatKind.JsonSchema:
                    request.GenerationConfig.ResponseMimeType = "application/json";
                    if (output.JsonSchema != null)
                    {
                        request.GenerationConfig.ResponseJsonSchema = output.JsonSchema.Schema;
                    }
                    break;
            }
        }

        private static GeminiThinkingConfig? ConvertReasoning(ReasoningOptions? reasoning)
        {
            if (reasoning == null)
                return null;
            if (reasoning.Effort == null &&
                reasoning.Output == null &&
                reasoning.Gemini?.ThinkingBudget == null)
                return null;

            var config = new GeminiThinkingConfig
            {
                IncludeThoughts = reasoning.Output switch
                {
                    ReasoningOutput.Summary => true,
                    ReasoningOutput.Omitted => false,
                    _ => null
                },
                ThinkingBudget = reasoning.Gemini?.ThinkingBudget
            };

            if (reasoning.Effort != null)
                config.ThinkingLevel = ToThinkingLevel(reasoning.Effort.Value).ToUpperInvariant();

            return config;
        }

        private static string ToThinkingLevel(ReasoningEffort effort) => effort switch
        {
            ReasoningEffort.None => "none",
            ReasoningEffort.Minimal => "minimal",
            ReasoningEffort.Low => "low",
            ReasoningEffort.Medium => "medium",
            ReasoningEffort.High => "high",
            ReasoningEffort.XHigh => "xhigh",
            ReasoningEffort.Max => "max",
            _ => throw new ArgumentOutOfRangeException(nameof(effort), effort, null)
        };

        private static void AddMediaPart(List<GeminiPart> parts, MediaContent media)
        {
            if (!string.IsNullOrEmpty(media.Source.Base64Data))
            {
                parts.Add(new GeminiPart
                {
                    InlineData = new GeminiInlineData
                    {
                        MimeType = media.MediaType,
                        Data = media.Source.Base64Data
                    }
                });
                return;
            }

            if (!string.IsNullOrEmpty(media.Source.Url) && string.IsNullOrEmpty(media.Source.FileUri))
            {
                throw new NotSupportedException(
                    "Gemini media URLs are not interchangeable with Gemini Files API URIs. Set MediaSource.FileUri.");
            }

            var fileUri = media.Source.FileUri;
            if (!string.IsNullOrEmpty(fileUri))
            {
                parts.Add(new GeminiPart
                {
                    FileData = new GeminiFileData
                    {
                        MimeType = media.MediaType,
                        FileUri = fileUri,
                        DisplayName = media.Source.FileName
                    }
                });
                return;
            }

            throw new NotSupportedException("Gemini media content requires base64 data or a Gemini file URI.");
        }

        private static void ApplyGeminiProviderMetadata(
            GeminiPart part,
            Dictionary<string, object>? providerMetadata)
        {
            if (providerMetadata == null)
                return;

            if (providerMetadata.TryGetValue("gemini.thoughtSignature", out var signature) &&
                signature is string signatureText)
            {
                part.ThoughtSignature = signatureText;
            }
        }

        private static Dictionary<string, object> ConvertToolResultOutput(object? output)
        {
            if (output is Dictionary<string, object> dictionary)
                return dictionary;

            return new Dictionary<string, object>
            {
                { "result", output ?? string.Empty }
            };
        }

        private FinishReason ConvertFinishReason(string? reason)
        {
            return reason switch
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
                _ => FinishReason.Other
            };
        }
    }
}
