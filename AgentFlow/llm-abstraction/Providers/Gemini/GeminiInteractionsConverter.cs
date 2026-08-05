using System.Text.Json;
using System.Text.Json.Serialization;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Providers.Gemini.Models;

namespace LLMAbstraction.Providers.Gemini
{
    /// <summary>
    /// Converts the unified contract to Gemini's recommended Interactions API.
    /// </summary>
    public sealed class GeminiInteractionsConverter
        : IModelConverter<GeminiInteractionRequest, GeminiInteractionResponse>
    {
        private static readonly JsonSerializerOptions NativeJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        private static readonly HashSet<string> ProtectedRequestFields = new(StringComparer.Ordinal)
        {
            "model", "input", "system_instruction", "tools", "tool_choice",
            "response_format", "stream", "store", "previous_interaction_id",
            "generation_config", "safety_settings"
        };

        public GeminiInteractionRequest ConvertRequest(UnifiedRequest request)
        {
            LLMRequestValidator.ValidateAndThrow(request, LLMProvider.Gemini);

            var result = new GeminiInteractionRequest
            {
                Model = request.Model,
                SystemInstruction = UnifiedRequestNormalization.CombineInstructions(request),
                Store = ReadStoreOption(request),
                GenerationConfig = ConvertGenerationConfig(request),
                ResponseFormat = ConvertResponseFormat(request.ResponseFormat)
            };

            if (request.Continuation?.Provider == ProviderIds.Gemini)
            {
                if (request.Continuation.Mode == ContinuationMode.ServerManaged &&
                    !string.IsNullOrWhiteSpace(request.Continuation.ResponseId))
                {
                    result.PreviousInteractionId = request.Continuation.ResponseId;
                    result.Store = true;
                }
                else
                {
                    result.Input.AddRange(request.Continuation.NativeItems.Select(item => item.Clone()));
                }
            }

            result.Input.AddRange(ConvertMessages(request.Messages));
            result.Tools = ConvertTools(request.Tools);

            if (request.ProviderOptions?.Gemini?.TryGetValue("safetySettings", out var safety) == true)
                result.SafetySettings = safety;

            result.AdditionalProperties = ProviderOptionMerger.ConvertAdditionalFields(
                request.ProviderOptions?.Gemini,
                ProtectedRequestFields,
                new HashSet<string>(StringComparer.Ordinal) { "safetySettings", "store" });

            return result;
        }

        public UnifiedResponse ConvertResponse(GeminiInteractionResponse response)
        {
            var content = new List<ContentBlock>();
            foreach (var step in response.Steps)
                ProjectStep(step, content);

            var message = new UnifiedMessage(MessageRole.Assistant, content);
            var finishReason = message.IsAssistantWithToolCalls()
                ? FinishReason.ToolCalls
                : ConvertStatus(response.Status);

            return new UnifiedResponse
            {
                Id = response.Id,
                Model = response.Model,
                Choices = new List<ResponseChoice>
                {
                    new()
                    {
                        Index = 0,
                        Message = message,
                        FinishReason = finishReason,
                        ProviderMetadata = new Dictionary<string, object>
                        {
                            ["gemini.status"] = response.Status ?? string.Empty
                        }
                    }
                },
                Usage = ConvertUsage(response.Usage),
                ProviderMetadata = new Dictionary<string, object>
                {
                    ["gemini.status"] = response.Status ?? string.Empty,
                    ["gemini.totalToolUseTokens"] = response.Usage?.TotalToolUseTokens ?? 0
                },
                Continuation = new ProviderContinuationState
                {
                    Provider = ProviderIds.Gemini,
                    Mode = ContinuationMode.StatelessReplay,
                    ResponseId = response.Id,
                    NativeItems = response.Steps.Select(step => step.Clone()).ToList()
                }
            };
        }

        internal static void ProjectStep(JsonElement step, List<ContentBlock> content)
        {
            var type = GetString(step, "type") ?? "unknown";
            var native = ProviderNativeRepresentation.Create(ProviderIds.Gemini, step);

            if (type == "model_output" && step.TryGetProperty("content", out var outputContent))
            {
                foreach (var item in outputContent.EnumerateArray())
                {
                    if (GetString(item, "type") == "text")
                    {
                        var text = new TextContent
                        {
                            Text = GetString(item, "text") ?? string.Empty,
                            NativeRepresentation = ProviderNativeRepresentation.Create(ProviderIds.Gemini, item)
                        };
                        if (item.TryGetProperty("annotations", out var annotations) &&
                            annotations.ValueKind == JsonValueKind.Array)
                        {
                            foreach (var annotation in annotations.EnumerateArray())
                            {
                                if (GetString(annotation, "type") != "url_citation")
                                    continue;
                                text.Citations.Add(new Citation
                                {
                                    Url = GetString(annotation, "url"),
                                    Title = GetString(annotation, "title"),
                                    StartIndex = GetInt(annotation, "start_index"),
                                    EndIndex = GetInt(annotation, "end_index"),
                                    NativeRepresentation = ProviderNativeRepresentation.Create(ProviderIds.Gemini, annotation)
                                });
                            }
                        }
                        content.Add(text);
                    }
                    else
                    {
                        content.Add(new ProviderNativeContent
                        {
                            NativeType = GetString(item, "type") ?? "unknown",
                            NativeRepresentation = ProviderNativeRepresentation.Create(ProviderIds.Gemini, item)
                        });
                    }
                }
                return;
            }

            if (type == "function_call")
            {
                content.Add(new ToolCallContent
                {
                    Id = GetString(step, "id") ?? Guid.NewGuid().ToString(),
                    Name = GetString(step, "name") ?? string.Empty,
                    Input = step.TryGetProperty("arguments", out var arguments)
                        ? JsonSerializer.Deserialize<Dictionary<string, object>>(arguments.GetRawText()) ?? new()
                        : new(),
                    NativeRepresentation = native
                });
                return;
            }

            if (type == "google_search_call")
            {
                content.Add(new ProviderToolCallContent
                {
                    Id = GetString(step, "id") ?? string.Empty,
                    ToolId = "web_search",
                    Capability = ProviderToolCapability.WebSearch,
                    Status = "completed",
                    Input = GetObject(step, "arguments"),
                    NativeRepresentation = native
                });
                return;
            }

            if (type == "google_search_result")
            {
                var result = step.TryGetProperty("result", out var resultValue)
                    ? resultValue
                    : default;
                content.Add(new ProviderToolResultContent
                {
                    ToolCallId = GetString(step, "call_id") ?? string.Empty,
                    Capability = ProviderToolCapability.WebSearch,
                    Status = "completed",
                    Output = result.ValueKind == JsonValueKind.Undefined
                        ? null
                        : JsonSerializer.Deserialize<object>(result.GetRawText()),
                    SearchSuggestionsHtml = GetString(result, "search_suggestions"),
                    Sources = ExtractSources(result),
                    NativeRepresentation = native
                });
                return;
            }

            content.Add(new ProviderNativeContent
            {
                NativeType = type,
                NativeRepresentation = native
            });
        }

        internal static UsageInfo ConvertUsage(GeminiInteractionUsage? usage) => new()
        {
            InputTokens = usage?.TotalInputTokens ?? 0,
            OutputTokens = usage?.TotalOutputTokens ?? 0,
            TotalTokens = usage?.TotalTokens ?? 0,
            ReasoningTokens = usage?.TotalThoughtTokens,
            CacheReadTokens = usage?.TotalCachedTokens,
            ProviderMetadata = usage?.TotalToolUseTokens != null
                ? new Dictionary<string, object>
                {
                    ["gemini.totalToolUseTokens"] = usage.TotalToolUseTokens.Value
                }
                : null
        };

        private static List<JsonElement> ConvertMessages(IEnumerable<UnifiedMessage> messages)
        {
            var steps = new List<JsonElement>();
            foreach (var message in messages.Where(message => message.Role != MessageRole.System))
            {
                var regularContent = new List<object>();
                foreach (var block in message.Content)
                {
                    if (block is ToolCallContent call)
                    {
                        FlushContentStep(message.Role, regularContent, steps);
                        steps.Add(ToElement(new Dictionary<string, object?>
                        {
                            ["type"] = "function_call",
                            ["id"] = call.Id,
                            ["name"] = call.Name,
                            ["arguments"] = call.Input
                        }));
                    }
                    else if (block is ToolResultContent result)
                    {
                        FlushContentStep(message.Role, regularContent, steps);
                        steps.Add(ToElement(new Dictionary<string, object?>
                        {
                            ["type"] = "function_result",
                            ["call_id"] = result.ToolCallId,
                            ["name"] = result.ToolName,
                            ["result"] = result.Output ?? string.Empty,
                            ["is_error"] = result.IsError
                        }));
                    }
                    else
                    {
                        regularContent.Add(ConvertContent(block));
                    }
                }
                FlushContentStep(message.Role, regularContent, steps);
            }
            return steps;
        }

        private static void FlushContentStep(
            MessageRole role,
            List<object> content,
            List<JsonElement> steps)
        {
            if (content.Count == 0)
                return;

            steps.Add(ToElement(new Dictionary<string, object?>
            {
                ["type"] = role == MessageRole.Assistant ? "model_output" : "user_input",
                ["content"] = content.ToArray()
            }));
            content.Clear();
        }

        private static object ConvertContent(ContentBlock block)
        {
            return block switch
            {
                TextContent text => new Dictionary<string, object?>
                {
                    ["type"] = "text",
                    ["text"] = text.Text
                },
                ImageContent image when image.Source.Data != null => new Dictionary<string, object?>
                {
                    ["type"] = "image",
                    ["mime_type"] = image.Source.MediaType,
                    ["data"] = image.Source.Data
                },
                ImageContent image when image.Source.Url != null => new Dictionary<string, object?>
                {
                    ["type"] = "image",
                    ["uri"] = image.Source.Url
                },
                MediaContent media when media.Source.Base64Data != null => new Dictionary<string, object?>
                {
                    ["type"] = media.MediaType.StartsWith("image/", StringComparison.Ordinal) ? "image" : "document",
                    ["mime_type"] = media.MediaType,
                    ["data"] = media.Source.Base64Data
                },
                MediaContent media when media.Source.FileUri != null => new Dictionary<string, object?>
                {
                    ["type"] = media.MediaType.StartsWith("image/", StringComparison.Ordinal) ? "image" : "document",
                    ["mime_type"] = media.MediaType,
                    ["uri"] = media.Source.FileUri
                },
                ProviderNativeContent native when native.NativeRepresentation?.Provider == ProviderIds.Gemini
                    => JsonSerializer.Deserialize<object>(native.NativeRepresentation.Value.GetRawText())!,
                _ => throw new NotSupportedException(
                    $"Content type '{block.Type}' cannot be represented by Gemini Interactions.")
            };
        }

        private static List<JsonElement>? ConvertTools(ToolCollection? tools)
        {
            var converted = tools?.Select(tool => tool switch
            {
                FunctionTool function => ToElement(new Dictionary<string, object?>
                {
                    ["type"] = "function",
                    ["name"] = function.Name,
                    ["description"] = function.Description,
                    ["parameters"] = function.Parameters
                }),
                ProviderTool { Capability: ProviderToolCapability.WebSearch } provider
                    => ConvertWebSearchTool((WebSearchOptions)provider.Options),
                ProviderTool provider => throw new NotSupportedException(
                    $"Gemini provider tool '{provider.Capability}' is not implemented."),
                _ => throw new NotSupportedException($"Unknown tool type '{tool.GetType().Name}'.")
            })
                .ToList();
            return converted?.Count > 0 ? converted : null;
        }

        private static JsonElement ConvertWebSearchTool(WebSearchOptions options)
        {
            var searchTypes = new List<string>();
            if (options.ContentTypes.HasFlag(WebSearchContentTypes.Web))
                searchTypes.Add("web_search");
            if (options.ContentTypes.HasFlag(WebSearchContentTypes.Image))
                searchTypes.Add("image_search");

            var definition = new Dictionary<string, object?>
            {
                ["type"] = "google_search"
            };
            if (searchTypes.Count > 0)
                definition["search_types"] = searchTypes;
            return ToElement(definition);
        }

        private static Dictionary<string, object>? ConvertGenerationConfig(UnifiedRequest request)
        {
            var config = new Dictionary<string, object>();
            Add(config, "max_output_tokens", request.Parameters.MaxOutputTokens);
            Add(config, "temperature", request.Parameters.Temperature);
            Add(config, "top_p", request.Parameters.TopP);
            Add(config, "top_k", request.Parameters.TopK);
            Add(config, "stop_sequences", request.Parameters.StopSequences);
            Add(config, "thinking_level", request.Reasoning?.Effort?.ToLowerInvariant());
            Add(config, "thinking_summaries", request.Reasoning?.IncludeThoughts == true ? "auto" : null);
            Add(config, "tool_choice", ConvertToolChoice(request.ToolChoice, request.Tools));
            return config.Count > 0 ? config : null;
        }

        private static List<JsonElement>? ConvertResponseFormat(ResponseFormat? format)
        {
            if (format == null || format.Type == ResponseFormatType.Text)
                return null;

            var definition = new Dictionary<string, object?>
            {
                ["type"] = "text",
                ["mime_type"] = "application/json"
            };
            if (format.Type == ResponseFormatType.JsonSchema && format.JsonSchema != null)
                definition["schema"] = format.JsonSchema.Schema;
            return new List<JsonElement> { ToElement(definition) };
        }

        private static object? ConvertToolChoice(ToolChoice? choice, ToolCollection? tools)
        {
            if (choice == null)
                return null;
            var mode = choice.Type switch
            {
                ToolChoiceType.Auto => "auto",
                ToolChoiceType.None => "none",
                ToolChoiceType.Required => "any",
                ToolChoiceType.Specific => "any",
                _ => "auto"
            };
            if (choice.Type != ToolChoiceType.Specific || string.IsNullOrWhiteSpace(choice.ToolName))
                return mode;
            var nativeName = tools?.OfType<ProviderTool>().Any(tool =>
                tool.Id == choice.ToolName && tool.Capability == ProviderToolCapability.WebSearch) == true
                    ? "google_search"
                    : choice.ToolName;
            return new Dictionary<string, object>
            {
                ["allowed_tools"] = new Dictionary<string, object>
                {
                    ["mode"] = mode,
                    ["tools"] = new[] { nativeName }
                }
            };
        }

        private static bool ReadStoreOption(UnifiedRequest request)
        {
            if (request.ProviderOptions?.Gemini?.TryGetValue("store", out var value) == true &&
                value is bool store)
                return store;
            return false;
        }

        private static FinishReason ConvertStatus(string? status) => status switch
        {
            "completed" => FinishReason.Stop,
            "requires_action" => FinishReason.ToolCalls,
            "failed" or "cancelled" => FinishReason.Error,
            "incomplete" => FinishReason.MaxTokens,
            _ => FinishReason.Other
        };

        private static JsonElement ToElement(object value) =>
            JsonSerializer.SerializeToElement(value, NativeJsonOptions);

        private static string? GetString(JsonElement element, string property) =>
            element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.String
                ? value.GetString()
                : null;

        private static int? GetInt(JsonElement element, string property) =>
            element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(property, out var value) && value.TryGetInt32(out var result)
                ? result
                : null;

        private static object? GetObject(JsonElement element, string property) =>
            element.ValueKind == JsonValueKind.Object && element.TryGetProperty(property, out var value)
                ? JsonSerializer.Deserialize<object>(value.GetRawText())
                : null;

        private static List<WebSource> ExtractSources(JsonElement result)
        {
            var sources = new List<WebSource>();
            if (result.ValueKind != JsonValueKind.Object ||
                !result.TryGetProperty("sources", out var sourceValues) ||
                sourceValues.ValueKind != JsonValueKind.Array)
                return sources;
            foreach (var source in sourceValues.EnumerateArray())
            {
                sources.Add(new WebSource
                {
                    Url = GetString(source, "url") ?? GetString(source, "uri"),
                    Title = GetString(source, "title"),
                    Snippet = GetString(source, "snippet"),
                    SourceType = GetString(source, "type")
                });
            }
            return sources;
        }

        private static void Add(Dictionary<string, object> target, string key, object? value)
        {
            if (value != null)
                target[key] = value;
        }
    }
}
