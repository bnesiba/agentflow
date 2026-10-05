using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Core;
using LLMAbstraction.Providers.OpenAI.Models;

namespace LLMAbstraction.Providers.OpenAI
{
    /// <summary>
    /// Converts between unified models and OpenAI Responses API models.
    /// </summary>
    public class OpenAIConverter : IModelConverter<OpenAIResponseRequest, OpenAIResponse>
    {
        private static readonly HashSet<string> ProtectedRequestFields = new(StringComparer.Ordinal)
        {
            "model", "input", "previous_response_id", "instructions", "max_output_tokens",
            "temperature", "top_p", "stream", "tools", "tool_choice", "parallel_tool_calls",
            "text", "reasoning", "user", "metadata", "include", "prompt_cache_key",
            "prompt_cache_options", "prompt_cache_retention"
        };
        private static readonly JsonSerializerOptions NativeJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
        private static readonly HashSet<string> SeparatelyHandledOptionFields = new(StringComparer.Ordinal)
        {
            "include"
        };

        public OpenAIResponseRequest ConvertRequest(UnifiedRequest request)
        {
            LLMRequestValidator.ValidateAndThrow(
                request,
                LLMProvider.OpenAI,
                LLMApiSurface.OpenAIResponses);
            var openAIRequest = new OpenAIResponseRequest
            {
                Model = request.Model,
                Input = ConvertInput(request),
                Instructions = UnifiedRequestNormalization.CombineInstructions(request),
                MaxOutputTokens = request.Parameters.MaxOutputTokens,
                Temperature = request.Parameters.Temperature,
                TopP = request.Parameters.TopP,
                Stream = request.Parameters.Stream,
                User = request.Metadata?.UserId,
                Metadata = request.Metadata?.Tags,
                PromptCacheKey = request.Cache?.OpenAI?.CacheKey,
                PromptCacheOptions = ConvertPromptCacheOptions(request),
                PromptCacheRetention = request.Cache?.OpenAI?.Retention switch
                {
                    OpenAIPromptCacheRetention.InMemory => "in_memory",
                    OpenAIPromptCacheRetention.TwentyFourHours => "24h",
                    _ => null
                },
                AdditionalProperties = ProviderOptionMerger.ConvertAdditionalFields(
                    request.ProviderOptions?.OpenAI,
                    ProtectedRequestFields,
                    SeparatelyHandledOptionFields),
                Include = ReadIncludeOptions(request.ProviderOptions?.OpenAI)
            };

            if (request.Continuation?.Provider == ProviderIds.OpenAI &&
                request.Continuation.Mode == ContinuationMode.ServerManaged)
            {
                openAIRequest.PreviousResponseId = request.Continuation.ResponseId;
            }

            var convertedTools = request.Tools?.Select(ConvertTool).ToList();
            if (convertedTools?.Count > 0)
            {
                openAIRequest.Tools = convertedTools;
                openAIRequest.ToolChoice = ConvertToolChoice(request.ToolChoice, request.Tools);

                if (request.Tools!.OfType<ProviderTool>().Any(tool =>
                    tool.Capability == ProviderToolCapability.WebSearch &&
                    ((WebSearchOptions)tool.Options).OpenAI?.IncludeAllSources == true))
                {
                    openAIRequest.Include ??= new List<string>();
                    if (!openAIRequest.Include.Contains("web_search_call.action.sources", StringComparer.Ordinal))
                        openAIRequest.Include.Add("web_search_call.action.sources");
                }

                if (request.Tools!.OfType<ProviderTool>().Any(tool =>
                    tool.Capability == ProviderToolCapability.WebSearch &&
                    ((WebSearchOptions)tool.Options).ContentTypes.HasFlag(WebSearchContentTypes.Image)))
                {
                    openAIRequest.Include ??= new List<string>();
                    if (!openAIRequest.Include.Contains("web_search_call.results", StringComparer.Ordinal))
                        openAIRequest.Include.Add("web_search_call.results");
                }

                if (request.ToolChoice?.DisableParallelToolUse != null)
                {
                    openAIRequest.ParallelToolCalls = !request.ToolChoice.DisableParallelToolUse.Value;
                }
            }

            if (request.Output != null)
            {
                openAIRequest.Text = new OpenAITextConfig
                {
                    Format = ConvertOutputFormat(request.Output)
                };
            }

            if (request.Reasoning != null)
            {
                openAIRequest.Reasoning = ConvertReasoning(request.Reasoning);
            }

            return openAIRequest;
        }

        public UnifiedResponse ConvertResponse(OpenAIResponse response)
        {
            var message = new UnifiedMessage
            {
                Role = MessageRole.Assistant,
                Content = new List<ContentBlock>()
            };

            foreach (var item in response.Output)
            {
                switch (item.Type)
                {
                    case "message":
                        AddMessageContent(message, item);
                        break;

                    case "function_call":
                        message.Content.Add(new ToolCallContent
                        {
                            Id = item.CallId ?? item.Id ?? string.Empty,
                            Name = item.Name ?? string.Empty,
                            Input = DeserializeArguments(item.Arguments),
                            NativeRepresentation = ProviderNativeRepresentation.Create(
                                ProviderIds.OpenAI,
                                JsonSerializer.SerializeToElement(item, NativeJsonOptions))
                        });
                        break;

                    case "web_search_call":
                        AddWebSearchContent(message, item);
                        break;
                }
            }

            EvidenceProjector.Project(message, ProviderIds.OpenAI);
            var finishReason = ConvertFinishReason(response, message);

            return new UnifiedResponse
            {
                Id = response.Id,
                Model = response.Model,
                Choices = new List<ResponseChoice>
                {
                    new ResponseChoice
                    {
                        Index = 0,
                        Message = message,
                        FinishReason = finishReason
                    }
                },
                Usage = response.Usage != null ? new UsageInfo
                {
                    InputTokens = response.Usage.InputTokens,
                    OutputTokens = response.Usage.OutputTokens,
                    TotalTokens = response.Usage.TotalTokens,
                    CacheReadTokens = response.Usage.InputTokenDetails?.CachedTokens,
                    CacheWriteTokens = response.Usage.InputTokenDetails?.CacheWriteTokens,
                    ReasoningTokens = response.Usage.OutputTokenDetails?.ReasoningTokens
                } : new UsageInfo(),
                ProviderMetadata = new Dictionary<string, object>
                {
                    { "status", response.Status ?? string.Empty },
                    { "incompleteReason", response.IncompleteDetails?.Reason ?? string.Empty },
                    { "errorCode", response.Error?.Code ?? string.Empty },
                    { "errorMessage", response.Error?.Message ?? string.Empty }
                },
                Continuation = CreateContinuationState(response)
            };
        }

        internal static ProviderContinuationState CreateContinuationState(OpenAIResponse response)
        {
            return new ProviderContinuationState
            {
                Provider = ProviderIds.OpenAI,
                ResponseId = response.Id,
                NativeItems = response.Output
                    .Select(item => JsonSerializer.SerializeToElement(item, NativeJsonOptions))
                    .Select(item => item.Clone())
                    .ToList()
            };
        }

        internal static void AddWebSearchContent(UnifiedMessage message, OpenAIOutputItem item)
        {
            var native = ProviderNativeRepresentation.Create(
                ProviderIds.OpenAI,
                JsonSerializer.SerializeToElement(item, NativeJsonOptions));
            var action = default(JsonElement);
            item.ExtensionData?.TryGetValue("action", out action);

            message.Content.Add(new ProviderToolCallContent
            {
                Id = item.Id ?? string.Empty,
                ToolId = "web_search",
                Capability = ProviderToolCapability.WebSearch,
                Status = item.Status,
                Input = action.ValueKind == JsonValueKind.Undefined
                    ? null
                    : JsonSerializer.Deserialize<object>(action.GetRawText()),
                NativeRepresentation = native
            });

            var sources = ExtractWebSources(action);
            if (item.ExtensionData?.TryGetValue("results", out var results) == true)
                sources.AddRange(ExtractWebSources(results));
            if (item.Status == "completed" || sources.Count > 0)
            {
                message.Content.Add(new ProviderToolResultContent
                {
                    ToolCallId = item.Id ?? string.Empty,
                    Capability = ProviderToolCapability.WebSearch,
                    Status = item.Status,
                    Output = action.ValueKind == JsonValueKind.Undefined
                        ? null
                        : JsonSerializer.Deserialize<object>(action.GetRawText()),
                    Sources = sources,
                    NativeRepresentation = native
                });
            }
        }

        private static List<WebSource> ExtractWebSources(JsonElement action)
        {
            var result = new List<WebSource>();
            if (action.ValueKind == JsonValueKind.Array)
            {
                AddWebSources(result, action);
                return result;
            }
            if (action.ValueKind != JsonValueKind.Object)
                return result;

            foreach (var property in new[] { "sources", "results" })
            {
                if (!action.TryGetProperty(property, out var sources) || sources.ValueKind != JsonValueKind.Array)
                    continue;
                AddWebSources(result, sources);
            }
            return result;
        }

        private static void AddWebSources(List<WebSource> target, JsonElement sources)
        {
            foreach (var source in sources.EnumerateArray())
            {
                target.Add(new WebSource
                {
                    Url = GetString(source, "url") ?? GetString(source, "source_website_url"),
                    Title = GetString(source, "title") ?? GetString(source, "caption"),
                    Snippet = GetString(source, "snippet"),
                    SourceType = GetString(source, "type"),
                    ImageUrl = GetString(source, "image_url"),
                    ThumbnailUrl = GetString(source, "thumbnail_url"),
                    Caption = GetString(source, "caption")
                });
            }
        }

        private List<object> ConvertInput(UnifiedRequest request)
        {
            var result = new List<object>();

            if (request.Continuation?.Provider == ProviderIds.OpenAI &&
                request.Continuation.Mode == ContinuationMode.StatelessReplay)
            {
                result.AddRange(request.Continuation.NativeItems.Select(item => (object)item.Clone()));
            }

            result.AddRange(ConvertMessages(request.Messages));
            return result;
        }

        private List<object> ConvertMessages(List<UnifiedMessage> messages)
        {
            var result = new List<object>();

            foreach (var message in messages)
            {
                if (message.Role == MessageRole.System)
                    continue;

                var messageContent = new List<object>();

                foreach (var block in message.Content)
                {
                    switch (block)
                    {
                        case TextContent text:
                            var textItem = new Dictionary<string, object?>
                            {
                                { "type", "input_text" },
                                { "text", text.Text }
                            };
                            AddPromptCacheBreakpoint(textItem, text.Cache);
                            messageContent.Add(textItem);
                            break;

                        case ImageContent image:
                            AddImageContent(messageContent, image.Source, image.Cache);
                            break;

                        case MediaContent media:
                            AddMediaContent(messageContent, media);
                            break;

                        case ToolCallContent toolCall:
                            result.Add(new Dictionary<string, object?>
                            {
                                { "type", "function_call" },
                                { "call_id", toolCall.Id },
                                { "name", toolCall.Name },
                                { "arguments", JsonSerializer.Serialize(toolCall.Input) }
                            });
                            break;

                        case ToolResultContent toolResult:
                            result.Add(new Dictionary<string, object?>
                            {
                                { "type", "function_call_output" },
                                { "call_id", toolResult.ToolCallId },
                                { "output", SerializeToolResult(toolResult.Output) }
                            });
                            break;

                        case ProviderNativeContent:
                            throw new NotSupportedException(
                                "Provider-native content cannot be sent to OpenAI unless it is supplied through OpenAI continuation state.");

                        default:
                            throw new NotSupportedException(
                                $"Content type '{block.Type}' is not supported by the OpenAI converter.");
                    }
                }

                if (messageContent.Count > 0)
                {
                    result.Add(new Dictionary<string, object?>
                    {
                        { "type", "message" },
                        { "role", ConvertRole(message.Role) },
                        { "content", messageContent }
                    });
                }
            }

            return result;
        }

        private static void AddImageContent(
            List<object> content,
            ImageSource source,
            PromptCacheDirective? cache)
        {
            Dictionary<string, object?> item;
            if (!string.IsNullOrEmpty(source.Url))
            {
                item = new Dictionary<string, object?>
                {
                    { "type", "input_image" },
                    { "detail", "auto" },
                    { "image_url", source.Url }
                };
            }
            else if (!string.IsNullOrEmpty(source.Data))
            {
                item = new Dictionary<string, object?>
                {
                    { "type", "input_image" },
                    { "detail", "auto" },
                    { "image_url", $"data:{source.MediaType ?? "image/jpeg"};base64,{source.Data}" }
                };
            }
            else
            {
                throw new NotSupportedException("OpenAI image content requires a URL or base64 data.");
            }
            AddPromptCacheBreakpoint(item, cache);
            content.Add(item);
        }

        private static void AddMediaContent(List<object> content, MediaContent media)
        {
            if (media.MediaType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
            {
                if (!string.IsNullOrEmpty(media.Source.FileId))
                {
                    var item = new Dictionary<string, object?>
                    {
                        { "type", "input_image" },
                        { "detail", "auto" },
                        { "file_id", media.Source.FileId }
                    };
                    AddPromptCacheBreakpoint(item, media.Cache);
                    content.Add(item);
                }
                else if (!string.IsNullOrEmpty(media.Source.Url))
                {
                    var item = new Dictionary<string, object?>
                    {
                        { "type", "input_image" },
                        { "detail", "auto" },
                        { "image_url", media.Source.Url }
                    };
                    AddPromptCacheBreakpoint(item, media.Cache);
                    content.Add(item);
                }
                else if (!string.IsNullOrEmpty(media.Source.Base64Data))
                {
                    var item = new Dictionary<string, object?>
                    {
                        { "type", "input_image" },
                        { "detail", "auto" },
                        { "image_url", $"data:{media.MediaType};base64,{media.Source.Base64Data}" }
                    };
                    AddPromptCacheBreakpoint(item, media.Cache);
                    content.Add(item);
                }
                else
                {
                    throw new NotSupportedException(
                        "OpenAI image media requires a file ID, URL, or base64 data. Provider file URIs are not accepted as image URLs.");
                }

                return;
            }

            if (!string.IsNullOrEmpty(media.Source.FileId))
            {
                var item = new Dictionary<string, object?>
                {
                    { "type", "input_file" },
                    { "file_id", media.Source.FileId }
                };
                AddPromptCacheBreakpoint(item, media.Cache);
                content.Add(item);
            }
            else if (!string.IsNullOrEmpty(media.Source.Url))
            {
                var item = new Dictionary<string, object?>
                {
                    { "type", "input_file" },
                    { "file_url", media.Source.Url }
                };
                AddPromptCacheBreakpoint(item, media.Cache);
                content.Add(item);
            }
            else if (!string.IsNullOrEmpty(media.Source.Base64Data))
            {
                var item = new Dictionary<string, object?>
                {
                    { "type", "input_file" },
                    { "filename", media.Source.FileName ?? "input" },
                    { "file_data", $"data:{media.MediaType};base64,{media.Source.Base64Data}" }
                };
                AddPromptCacheBreakpoint(item, media.Cache);
                content.Add(item);
            }
            else
            {
                throw new NotSupportedException(
                    "OpenAI media content requires a file ID, URL, or base64 data. Provider file URIs are not accepted as file URLs.");
            }
        }

        private static void AddPromptCacheBreakpoint(
            Dictionary<string, object?> item,
            PromptCacheDirective? directive)
        {
            if (directive != null)
                item["prompt_cache_breakpoint"] = new Dictionary<string, object> { ["mode"] = "explicit" };
        }

        private static OpenAIPromptCacheConfig? ConvertPromptCacheOptions(UnifiedRequest request)
        {
            var hasBreakpoints = request.Messages.SelectMany(message => message.Content)
                .Any(block => block.Cache != null);
            var mode = request.Cache?.Mode switch
            {
                PromptCacheMode.ExplicitBreakpointsOnly or PromptCacheMode.Disabled => "explicit",
                PromptCacheMode.PreferReuse => "implicit",
                _ when hasBreakpoints => "explicit",
                _ => null
            };
            var ttl = request.Cache?.Ttl == PromptCacheTtl.ThirtyMinutes ? "30m" : null;
            return mode != null || ttl != null
                ? new OpenAIPromptCacheConfig { Mode = mode, Ttl = ttl }
                : null;
        }

        private static string ConvertRole(MessageRole role)
        {
            return role switch
            {
                MessageRole.System => "developer",
                MessageRole.User => "user",
                MessageRole.Assistant => "assistant",
                MessageRole.Tool => "user",
                _ => throw new ArgumentException($"Unknown role: {role}")
            };
        }

        private static OpenAIResponseTool ConvertTool(LLMTool tool)
        {
            return tool switch
            {
                FunctionTool function => ConvertFunctionTool(function),
                ProviderTool { Capability: ProviderToolCapability.WebSearch } provider
                    => ConvertWebSearchTool((WebSearchOptions)provider.Options),
                ProviderTool provider => throw new NotSupportedException(
                    $"OpenAI provider tool '{provider.Capability}' is not implemented."),
                _ => throw new NotSupportedException($"Unknown tool type '{tool.GetType().Name}'.")
            };
        }

        private static OpenAIResponseTool ConvertFunctionTool(FunctionTool tool)
        {
            return new OpenAIResponseTool
            {
                Type = "function",
                Name = tool.Name,
                Description = tool.Description,
                Parameters = tool.Parameters,
                Strict = tool.Strict ?? true
            };
        }

        private static OpenAIResponseTool ConvertWebSearchTool(WebSearchOptions options)
        {
            var native = options.OpenAI;
            var filters = new Dictionary<string, object>();
            if (options.AllowedDomains?.Count > 0)
                filters["allowed_domains"] = options.AllowedDomains;
            if (options.BlockedDomains?.Count > 0)
                filters["blocked_domains"] = options.BlockedDomains;

            var contentTypes = new List<string>();
            if (options.ContentTypes.HasFlag(WebSearchContentTypes.Web))
                contentTypes.Add("text");
            if (options.ContentTypes.HasFlag(WebSearchContentTypes.Image))
                contentTypes.Add("image");

            Dictionary<string, object>? imageSettings = null;
            if (native?.MaximumImageResults != null || native?.IncludeImageCaptions == true)
            {
                imageSettings = new Dictionary<string, object>();
                if (native.MaximumImageResults != null)
                    imageSettings["max_results"] = native.MaximumImageResults.Value;
                if (native.IncludeImageCaptions)
                    imageSettings["caption"] = true;
            }

            return new OpenAIResponseTool
            {
                Type = "web_search",
                Filters = filters.Count > 0 ? filters : null,
                UserLocation = ConvertLocation(options.Location),
                SearchContextSize = native?.ContextSize?.ToString().ToLowerInvariant(),
                ReturnTokenBudget = native?.UnlimitedReturnTokenBudget == true ? "unlimited" : null,
                ExternalWebAccess = native != null ? native.ExternalWebAccess : null,
                SearchContentTypes = contentTypes.Count > 0 ? contentTypes : null,
                ImageSettings = imageSettings
            };
        }

        private static Dictionary<string, object>? ConvertLocation(ApproximateLocation? location)
        {
            if (location == null)
                return null;
            var result = new Dictionary<string, object> { ["type"] = "approximate" };
            if (location.City != null) result["city"] = location.City;
            if (location.Region != null) result["region"] = location.Region;
            if (location.Country != null) result["country"] = location.Country;
            if (location.Timezone != null) result["timezone"] = location.Timezone;
            return result;
        }

        private static List<string>? ReadIncludeOptions(Dictionary<string, object>? options)
        {
            if (options?.TryGetValue("include", out var value) != true)
                return null;
            if (value is IEnumerable<string> strings)
                return strings.ToList();
            if (value is JsonElement { ValueKind: JsonValueKind.Array } json)
                return json.EnumerateArray().Select(item => item.GetString() ?? string.Empty).ToList();
            throw new ArgumentException("OpenAI provider option 'include' must be a collection of strings.");
        }

        private static object? ConvertToolChoice(ToolChoice? toolChoice, ToolCollection? tools)
        {
            if (toolChoice == null)
                return null;

            return toolChoice.Type switch
            {
                ToolChoiceType.Auto => "auto",
                ToolChoiceType.None => "none",
                ToolChoiceType.Required => "required",
                ToolChoiceType.Specific when tools?.OfType<ProviderTool>().Any(tool =>
                    tool.Id == toolChoice.ToolName &&
                    tool.Capability == ProviderToolCapability.WebSearch) == true
                    => new Dictionary<string, object?> { { "type", "web_search" } },
                ToolChoiceType.Specific => new Dictionary<string, object?>
                {
                    { "type", "function" },
                    { "name", toolChoice.ToolName }
                },
                _ => "auto"
            };
        }

        private static object? ConvertOutputFormat(OutputFormat output)
        {
            return output.Kind switch
            {
                OutputFormatKind.Text => new Dictionary<string, object?>
                {
                    { "type", "text" }
                },
                OutputFormatKind.JsonObject => new Dictionary<string, object?>
                {
                    { "type", "json_object" }
                },
                OutputFormatKind.JsonSchema => new Dictionary<string, object?>
                {
                    { "type", "json_schema" },
                    { "name", output.JsonSchema!.Name },
                    { "description", output.JsonSchema.Description },
                    { "schema", output.JsonSchema.Schema },
                    { "strict", true }
                },
                _ => null
            };
        }

        private static OpenAIReasoningConfig? ConvertReasoning(ReasoningOptions reasoning)
        {
            if (reasoning.Effort == null &&
                reasoning.Output is null or ReasoningOutput.Omitted &&
                reasoning.OpenAI == null)
                return null;

            return new OpenAIReasoningConfig
            {
                Effort = reasoning.Effort == null ? null : ToWireValue(reasoning.Effort.Value),
                Summary = reasoning.OpenAI?.Summary switch
                {
                    OpenAIReasoningSummary.Auto => "auto",
                    OpenAIReasoningSummary.Concise => "concise",
                    OpenAIReasoningSummary.Detailed => "detailed",
                    null when reasoning.Output == ReasoningOutput.Summary => "auto",
                    null when reasoning.Output == ReasoningOutput.Omitted => null,
                    _ => null
                },
                Mode = reasoning.OpenAI?.Mode switch
                {
                    OpenAIReasoningMode.Standard => "standard",
                    OpenAIReasoningMode.Pro => "pro",
                    _ => null
                },
                Context = reasoning.OpenAI?.Context switch
                {
                    OpenAIReasoningContext.Auto => "auto",
                    OpenAIReasoningContext.CurrentTurn => "current_turn",
                    OpenAIReasoningContext.AllTurns => "all_turns",
                    _ => null
                }
            };
        }

        private static string ToWireValue(ReasoningEffort effort) => effort switch
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

        private static void AddMessageContent(UnifiedMessage message, OpenAIOutputItem item)
        {
            if (item.Content == null)
                return;

            foreach (var content in item.Content)
            {
                if ((content.Type == "output_text" || content.Type == "text") && content.Text != null)
                {
                    var text = new TextContent
                    {
                        Text = content.Text,
                        NativeRepresentation = ProviderNativeRepresentation.Create(
                            ProviderIds.OpenAI,
                            JsonSerializer.SerializeToElement(content, NativeJsonOptions))
                    };
                    if (content.ExtensionData?.TryGetValue("annotations", out var annotations) == true &&
                        annotations.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var annotation in annotations.EnumerateArray())
                        {
                            var annotationType = GetString(annotation, "type");
                            var annotationIndex = GetInt(annotation, "index");
                            text.Citations.Add(new Citation
                            {
                                Url = GetString(annotation, "url"),
                                Title = GetString(annotation, "title"),
                                FileId = GetString(annotation, "file_id"),
                                FileName = GetString(annotation, "filename"),
                                StartIndex = GetInt(annotation, "start_index"),
                                EndIndex = GetInt(annotation, "end_index"),
                                ProviderMetadata = annotationType != null || annotationIndex != null
                                    ? new Dictionary<string, object>
                                    {
                                        ["openai.annotationType"] = annotationType ?? string.Empty,
                                        ["openai.annotationIndex"] = annotationIndex ?? -1
                                    }
                                    : null,
                                NativeRepresentation = ProviderNativeRepresentation.Create(ProviderIds.OpenAI, annotation)
                            });
                        }
                    }
                    if (content.ExtensionData != null && content.ExtensionData.Count > 0)
                    {
                        text.ProviderMetadata = content.ExtensionData.ToDictionary(
                            pair => $"openai.{pair.Key}",
                            pair => (object)pair.Value.Clone());
                    }
                    message.Content.Add(text);
                }
                else if (content.Type == "refusal" && content.Refusal != null)
                {
                    message.Content.Add(new RefusalContent { Refusal = content.Refusal });
                }
            }
        }

        private static Dictionary<string, object> DeserializeArguments(string? arguments)
        {
            if (string.IsNullOrWhiteSpace(arguments))
                return new Dictionary<string, object>();

            try
            {
                return JsonSerializer.Deserialize<Dictionary<string, object>>(arguments)
                    ?? new Dictionary<string, object>();
            }
            catch
            {
                return new Dictionary<string, object>
                {
                    { "raw", arguments }
                };
            }
        }

        private static string SerializeToolResult(object? output)
        {
            return output switch
            {
                null => string.Empty,
                string text => text,
                _ => JsonSerializer.Serialize(output)
            };
        }

        private static string? GetString(JsonElement element, string property) =>
            element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(property, out var value) &&
            value.ValueKind == JsonValueKind.String
                ? value.GetString()
                : null;

        private static int? GetInt(JsonElement element, string property) =>
            element.ValueKind == JsonValueKind.Object &&
            element.TryGetProperty(property, out var value) &&
            value.TryGetInt32(out var number)
                ? number
                : null;

        private static FinishReason ConvertFinishReason(OpenAIResponse response, UnifiedMessage message)
        {
            if (message.Content.Any(block => block is ToolCallContent))
                return FinishReason.ToolCalls;

            if (message.Content.Any(block => block is RefusalContent))
                return FinishReason.ContentFilter;

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
