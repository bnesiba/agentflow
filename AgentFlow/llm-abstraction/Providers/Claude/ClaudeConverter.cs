using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Core;
using LLMAbstraction.Providers.Claude.Models;

namespace LLMAbstraction.Providers.Claude
{
    /// <summary>
    /// Converts between unified models and Claude-specific models
    /// </summary>
    public class ClaudeConverter : IModelConverter<ClaudeMessageRequest, ClaudeMessageResponse>
    {
        private static readonly HashSet<string> ProtectedRequestFields = new(StringComparer.Ordinal)
        {
            "model", "max_tokens", "messages", "system", "temperature", "top_p", "top_k",
            "stop_sequences", "stream", "tools", "tool_choice", "output_config", "thinking", "metadata",
            "cache_control"
        };
        private static readonly HashSet<string> SeparatelyHandledOptionFields = new(StringComparer.Ordinal)
        {
            "anthropicBeta"
        };
        private static readonly JsonSerializerOptions NativeJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        public ClaudeMessageRequest ConvertRequest(UnifiedRequest request)
        {
            LLMRequestValidator.ValidateAndThrow(
                request,
                LLMProvider.Claude,
                LLMApiSurface.AnthropicMessages);
            var thinking = ConvertReasoning(request.Reasoning);

            var claudeRequest = new ClaudeMessageRequest
            {
                Model = request.Model,
                MaxTokens = request.Parameters.MaxOutputTokens ?? 1024, // Claude requires max_tokens
                Messages = ConvertMessages(request.Messages),
                System = ConvertSystem(request),
                CacheControl = request.Cache?.Mode == PromptCacheMode.PreferReuse
                    ? ConvertCacheControl(request.Cache.Ttl)
                    : null,
                Temperature = SupportsTemperature(thinking) ? request.Parameters.Temperature : null,
                TopP = SupportsTopP(thinking, request.Parameters.TopP) ? request.Parameters.TopP : null,
                TopK = SupportsTopK(thinking) ? request.Parameters.TopK : null,
                StopSequences = request.Parameters.StopSequences,
                Stream = request.Parameters.Stream,
                Thinking = thinking,
                Metadata = request.Metadata?.UserId != null
                    ? new ClaudeMetadata { UserId = request.Metadata.UserId }
                    : null,
                AdditionalProperties = ProviderOptionMerger.ConvertAdditionalFields(
                    request.ProviderOptions?.Claude,
                    ProtectedRequestFields,
                    SeparatelyHandledOptionFields)
            };

            // Convert tools if present
            var convertedTools = request.Tools?.Select(ConvertTool).ToList();
            if (convertedTools?.Count > 0)
            {
                claudeRequest.Tools = convertedTools;
                claudeRequest.ToolChoice = ConvertToolChoice(request.ToolChoice, request.Tools);
            }

            claudeRequest.OutputConfig = ConvertOutputConfig(request.Output, request.Reasoning);

            return claudeRequest;
        }

        private static bool SupportsTemperature(ClaudeThinkingConfig? thinking)
        {
            return !IsThinkingEnabled(thinking);
        }

        private static bool SupportsTopK(ClaudeThinkingConfig? thinking)
        {
            return !IsThinkingEnabled(thinking);
        }

        private static bool SupportsTopP(ClaudeThinkingConfig? thinking, double? topP)
        {
            return !IsThinkingEnabled(thinking) || topP == null || topP >= 0.95 && topP <= 1.0;
        }

        private static bool IsThinkingEnabled(ClaudeThinkingConfig? thinking)
        {
            return thinking?.Type == "enabled" || thinking?.Type == "adaptive";
        }

        public UnifiedResponse ConvertResponse(ClaudeMessageResponse response)
        {
            var message = ConvertMessage(response);
            EvidenceProjector.Project(message, ProviderIds.Claude);
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
                        FinishReason = ConvertFinishReason(response.StopReason)
                    }
                },
                Usage = new UsageInfo
                {
                    InputTokens = response.Usage.InputTokens,
                    OutputTokens = response.Usage.OutputTokens,
                    TotalTokens = response.Usage.InputTokens + response.Usage.OutputTokens,
                    CacheCreationTokens = response.Usage.CacheCreationInputTokens,
                    CacheWriteTokens = response.Usage.CacheCreationInputTokens,
                    CacheWrite5MinuteTokens = response.Usage.CacheCreation?.Ephemeral5mInputTokens ?? response.Usage.Ephemeral5mInputTokens,
                    CacheWrite1HourTokens = response.Usage.CacheCreation?.Ephemeral1hInputTokens ?? response.Usage.Ephemeral1hInputTokens,
                    CacheReadTokens = response.Usage.CacheReadInputTokens,
                    ProviderMetadata = response.Usage.ServerToolUse != null
                        ? new Dictionary<string, object>
                        {
                            ["claude.serverToolUse"] = response.Usage.ServerToolUse
                        }
                        : null
                },
                ProviderMetadata = new Dictionary<string, object>
                {
                    ["claude.stopReason"] = response.StopReason ?? string.Empty,
                    ["claude.stopSequence"] = response.StopSequence ?? string.Empty,
                    ["claude.cacheCreationEphemeral5mTokens"] = response.Usage.Ephemeral5mInputTokens ?? 0,
                    ["claude.cacheCreationEphemeral1hTokens"] = response.Usage.Ephemeral1hInputTokens ?? 0
                }
            };
        }

        private List<ClaudeMessage> ConvertMessages(List<UnifiedMessage> messages)
        {
            var result = new List<ClaudeMessage>();

            foreach (var message in messages)
            {
                // Skip system messages (handled separately in Claude)
                if (message.Role == MessageRole.System)
                    continue;

                var role = message.Role switch
                {
                    MessageRole.User => "user",
                    MessageRole.Assistant => "assistant",
                    MessageRole.Tool => "user", // Tool results go in user messages in Claude
                    _ => throw new ArgumentException($"Unknown role: {message.Role}")
                };

                var content = ConvertContent(message.Content);

                result.Add(new ClaudeMessage
                {
                    Role = role,
                    Content = content
                });
            }

            return result;
        }

        private object ConvertContent(List<ContentBlock> content)
        {
            // If single text block, use string shorthand
            if (content.Count == 1 &&
                content[0] is TextContent textContent &&
                content[0].Cache == null &&
                !HasClaudeNativeRepresentation(textContent))
            {
                return textContent.Text;
            }

            // Otherwise, convert to array of content blocks
            var blocks = new List<object>();

            foreach (var block in content)
            {
                if (TryGetClaudeNativeValue(block, out var nativeValue))
                {
                    blocks.Add(nativeValue);
                    continue;
                }

                switch (block)
                {
                    case TextContent text:
                        blocks.Add(ApplyCacheControl(new
                        {
                            type = "text",
                            text = text.Text
                        }, block.Cache));
                        break;

                    case ImageContent image:
                        if (!string.IsNullOrEmpty(image.Source.Url))
                        {
                            blocks.Add(ApplyCacheControl(new
                            {
                                type = "image",
                                source = new
                                {
                                    type = "url",
                                    url = image.Source.Url
                                }
                            }, block.Cache));
                        }
                        else if (!string.IsNullOrEmpty(image.Source.Data))
                        {
                            blocks.Add(ApplyCacheControl(new
                            {
                                type = "image",
                                source = new
                                {
                                    type = "base64",
                                    media_type = image.Source.MediaType,
                                    data = image.Source.Data
                                }
                            }, block.Cache));
                        }
                        else
                        {
                            throw new NotSupportedException("Claude image content requires a URL or base64 data.");
                        }
                        break;

                    case MediaContent media:
                        blocks.Add(ApplyCacheControl(ConvertMediaContent(media), block.Cache));
                        break;

                    case SearchResultContent searchResult:
                        blocks.Add(ApplyCacheControl(ConvertSearchResult(searchResult), block.Cache));
                        break;

                    case ToolCallContent toolCall:
                        blocks.Add(ApplyCacheControl(new
                        {
                            type = "tool_use",
                            id = toolCall.Id,
                            name = toolCall.Name,
                            input = toolCall.Input
                        }, block.Cache));
                        break;

                    case ToolResultContent toolResult:
                        blocks.Add(ApplyCacheControl(new
                        {
                            type = "tool_result",
                            tool_use_id = toolResult.ToolCallId,
                            content = toolResult.Output,
                            is_error = toolResult.IsError
                        }, block.Cache));
                        break;

                    case ProviderNativeContent:
                        throw new NotSupportedException(
                            "Provider-native content cannot be sent to Claude unless it originated from Claude.");

                    default:
                        throw new NotSupportedException(
                            $"Content type '{block.Type}' is not supported by the Claude converter.");
                }
            }

            return blocks;
        }

        private static object? ConvertSystem(UnifiedRequest request)
        {
            var instructions = UnifiedRequestNormalization.CombineInstructions(request);
            if (instructions == null || request.InstructionsCache == null)
                return instructions;
            return new List<object>
            {
                ApplyCacheControl(new { type = "text", text = instructions }, request.InstructionsCache)
            };
        }

        private static object ApplyCacheControl(object value, PromptCacheDirective? directive)
        {
            if (directive == null)
                return value;
            var json = JsonSerializer.SerializeToElement(value, NativeJsonOptions);
            var result = json.EnumerateObject().ToDictionary(
                property => property.Name,
                property => (object?)property.Value.Clone());
            result["cache_control"] = ConvertCacheControl(directive.Ttl);
            return result;
        }

        private static ClaudeCacheControl ConvertCacheControl(PromptCacheTtl? ttl) => new()
        {
            Ttl = ttl switch
            {
                PromptCacheTtl.FiveMinutes => "5m",
                PromptCacheTtl.OneHour => "1h",
                _ => null
            }
        };

        private static object ConvertMediaContent(MediaContent media)
        {
            if (media is DocumentContent document)
            {
                var portableMedia = new MediaContent
                {
                    MediaType = document.MediaType,
                    Source = document.Source
                };
                var json = JsonSerializer.SerializeToElement(
                    ConvertMediaContent(portableMedia), NativeJsonOptions);
                var result = json.EnumerateObject().ToDictionary(
                    property => property.Name,
                    property => (object?)property.Value.Clone());
                if (document.Title != null) result["title"] = document.Title;
                if (document.Context != null) result["context"] = document.Context;
                if (document.CitationsEnabled != null)
                    result["citations"] = new Dictionary<string, object>
                    {
                        ["enabled"] = document.CitationsEnabled.Value
                    };
                return result;
            }

            var contentType = media.MediaType.StartsWith("image/", StringComparison.OrdinalIgnoreCase)
                ? "image"
                : media.MediaType == "application/pdf" || media.MediaType == "text/plain"
                    ? "document"
                    : throw new NotSupportedException(
                        $"Claude media type '{media.MediaType}' is not supported by this converter.");

            if (!string.IsNullOrEmpty(media.Source.FileId))
            {
                return new
                {
                    type = contentType,
                    source = new
                    {
                        type = "file",
                        file_id = media.Source.FileId
                    }
                };
            }

            if (!string.IsNullOrEmpty(media.Source.Url) || !string.IsNullOrEmpty(media.Source.FileUri))
            {
                return new
                {
                    type = contentType,
                    source = new
                    {
                        type = "url",
                        url = media.Source.Url ?? media.Source.FileUri
                    }
                };
            }

            if (!string.IsNullOrEmpty(media.Source.Base64Data))
            {
                return new
                {
                    type = contentType,
                    source = new
                    {
                        type = "base64",
                        media_type = media.MediaType,
                        data = media.Source.Base64Data
                    }
                };
            }

            throw new NotSupportedException("Claude media content requires a file ID, URL, file URI, or base64 data.");
        }

        private static object ConvertSearchResult(SearchResultContent result) =>
            new Dictionary<string, object?>
            {
                ["type"] = "search_result",
                ["source"] = result.Source,
                ["title"] = result.Title,
                ["content"] = result.Content.Select(text => new Dictionary<string, object>
                {
                    ["type"] = "text",
                    ["text"] = text.Text
                }).ToList(),
                ["citations"] = result.CitationsEnabled == null
                    ? null
                    : new Dictionary<string, object> { ["enabled"] = result.CitationsEnabled.Value }
            };

        private ClaudeTool ConvertTool(LLMTool tool)
        {
            return tool switch
            {
                FunctionTool function => ConvertFunctionTool(function),
                ProviderTool { Capability: ProviderToolCapability.WebSearch } provider
                    => ConvertWebSearchTool(provider),
                ProviderTool provider => throw new NotSupportedException(
                    $"Anthropic provider tool '{provider.Capability}' is not implemented."),
                _ => throw new NotSupportedException($"Unknown tool type '{tool.GetType().Name}'.")
            };
        }

        private ClaudeTool ConvertFunctionTool(FunctionTool tool)
        {
            return new ClaudeTool
            {
                Name = tool.Name,
                Description = tool.Description,
                InputSchema = tool.Parameters,
                Strict = tool.Strict,
                CacheControl = tool.Cache == null ? null : ConvertCacheControl(tool.Cache.Ttl)
            };
        }

        private static ClaudeTool ConvertWebSearchTool(ProviderTool tool)
        {
            var options = (WebSearchOptions)tool.Options;
            var native = options.Anthropic;
            var type = native?.IncludeFullResults != null
                ? "web_search_20260318"
                : native?.DynamicFiltering == true
                    ? "web_search_20260209"
                    : "web_search_20250305";

            return new ClaudeTool
            {
                Type = type,
                Name = "web_search",
                MaxUses = native?.MaximumUses,
                AllowedDomains = options.AllowedDomains,
                BlockedDomains = options.BlockedDomains,
                UserLocation = ConvertLocation(options.Location),
                AllowedCallers = native?.DynamicFiltering == false
                    ? new List<string> { "direct" }
                    : null,
                ResponseInclusion = native?.IncludeFullResults switch
                {
                    true => "full",
                    false => "excluded",
                    _ => null
                },
                CacheControl = tool.Cache == null ? null : ConvertCacheControl(tool.Cache.Ttl)
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

        private object? ConvertToolChoice(ToolChoice? toolChoice, ToolCollection? tools)
        {
            if (toolChoice == null)
                return null;

            return toolChoice.Type switch
            {
                ToolChoiceType.Auto => new { type = "auto" },
                ToolChoiceType.None => new { type = "none" },
                ToolChoiceType.Required => new { type = "any" },
                ToolChoiceType.Specific => new
                {
                    type = "tool",
                    name = tools?.OfType<ProviderTool>().Any(tool =>
                        tool.Id == toolChoice.ToolName &&
                        tool.Capability == ProviderToolCapability.WebSearch) == true
                            ? "web_search"
                            : toolChoice.ToolName
                },
                _ => new { type = "auto" }
            };
        }

        private UnifiedMessage ConvertMessage(ClaudeMessageResponse response)
        {
            var message = new UnifiedMessage
            {
                Role = MessageRole.Assistant,
                Content = new List<ContentBlock>()
            };

            foreach (var block in response.Content)
            {
                message.Content.Add(ConvertResponseContentBlock(block));
            }

            return message;
        }

        internal static ContentBlock ConvertResponseContentBlock(ClaudeContentBlock block)
        {
            var nativeRepresentation = ProviderNativeRepresentation.Create(
                ProviderIds.Claude,
                JsonSerializer.SerializeToElement(block, NativeJsonOptions));

            if (block.Type == "text")
            {
                var text = new TextContent
                {
                    Text = block.Text ?? string.Empty,
                    NativeRepresentation = nativeRepresentation
                };
                if (block.AdditionalProperties?.TryGetValue("citations", out var citations) == true &&
                    citations.ValueKind == JsonValueKind.Array)
                {
                    foreach (var citation in citations.EnumerateArray())
                        text.Citations.Add(ConvertCitation(citation));
                }
                return text;
            }

            if (block.Type == "tool_use")
            {
                return new ToolCallContent
                {
                    Id = block.Id ?? string.Empty,
                    Name = block.Name ?? string.Empty,
                    Input = block.Input ?? new Dictionary<string, object>(),
                    NativeRepresentation = nativeRepresentation
                };
            }

            if (block.Type == "server_tool_use" && block.Name == "web_search")
            {
                return new ProviderToolCallContent
                {
                    Id = block.Id ?? string.Empty,
                    ToolId = "web_search",
                    Capability = ProviderToolCapability.WebSearch,
                    Status = "completed",
                    Input = block.Input,
                    NativeRepresentation = nativeRepresentation
                };
            }

            if (block.Type == "web_search_tool_result")
            {
                var resultContent = default(JsonElement);
                block.AdditionalProperties?.TryGetValue("content", out resultContent);
                var isError = IsAnthropicToolError(resultContent);
                return new ProviderToolResultContent
                {
                    ToolCallId = block.ToolUseId ?? string.Empty,
                    Capability = ProviderToolCapability.WebSearch,
                    Status = isError == true ? "failed" : "completed",
                    Output = resultContent.ValueKind == JsonValueKind.Undefined
                        ? null
                        : JsonSerializer.Deserialize<object>(resultContent.GetRawText()),
                    IsError = isError,
                    Sources = ExtractAnthropicSources(resultContent),
                    NativeRepresentation = nativeRepresentation
                };
            }

            return new ProviderNativeContent
            {
                NativeType = block.Type,
                NativeRepresentation = nativeRepresentation
            };
        }

        private static bool HasClaudeNativeRepresentation(ContentBlock block)
        {
            return block.NativeRepresentation?.Provider == ProviderIds.Claude;
        }

        private static List<WebSource> ExtractAnthropicSources(JsonElement content)
        {
            var sources = new List<WebSource>();
            if (content.ValueKind != JsonValueKind.Array)
                return sources;
            foreach (var item in content.EnumerateArray())
            {
                if (GetString(item, "type") is not ("web_search_result" or "web_search_result_location"))
                    continue;
                sources.Add(new WebSource
                {
                    Url = GetString(item, "url"),
                    Title = GetString(item, "title"),
                    Snippet = GetString(item, "snippet"),
                    SourceType = GetString(item, "type"),
                    PageAge = GetString(item, "page_age")
                });
            }
            return sources;
        }

        private static bool? IsAnthropicToolError(JsonElement content)
        {
            if (content.ValueKind != JsonValueKind.Object)
                return false;
            return GetString(content, "type") == "web_search_tool_result_error";
        }

        private static Citation ConvertCitation(JsonElement citation)
        {
            var type = GetString(citation, "type");
            SourceLocation? location = type switch
            {
                "char_location" => CreateCharacterLocation(citation),
                "page_location" => CreatePageLocation(citation),
                "content_block_location" or "search_result_location" =>
                    CreateBlockLocation(citation),
                _ => null
            };
            var metadata = new Dictionary<string, object>();
            AddIntMetadata(metadata, citation, "document_index", "claude.documentIndex");
            AddIntMetadata(metadata, citation, "search_result_index", "claude.searchResultIndex");
            AddStringMetadata(metadata, citation, "source", "claude.source");
            AddStringMetadata(metadata, citation, "encrypted_index", "claude.encryptedIndex");

            return new Citation
            {
                Url = GetString(citation, "url"),
                Title = GetString(citation, "title") ?? GetString(citation, "document_title"),
                FileId = GetString(citation, "file_id"),
                CitedText = GetString(citation, "cited_text"),
                SourceLocation = location,
                ProviderMetadata = metadata.Count == 0 ? null : metadata,
                NativeRepresentation = ProviderNativeRepresentation.Create(ProviderIds.Claude, citation)
            };
        }

        private static CharacterRangeLocation? CreateCharacterLocation(JsonElement citation)
        {
            var start = GetInt(citation, "start_char_index");
            var end = GetInt(citation, "end_char_index");
            return start != null && end != null
                ? new CharacterRangeLocation
                {
                    StartIndex = start.Value,
                    EndIndex = end.Value,
                    Unit = TextIndexUnit.ProviderDefined,
                    EndExclusive = true
                }
                : null;
        }

        private static PageRangeLocation? CreatePageLocation(JsonElement citation)
        {
            var start = GetInt(citation, "start_page_number");
            var end = GetInt(citation, "end_page_number");
            return start != null && end != null
                ? new PageRangeLocation
                {
                    StartPage = start.Value,
                    EndPage = end.Value,
                    EndInclusive = null
                }
                : null;
        }

        private static ContentBlockRangeLocation? CreateBlockLocation(JsonElement citation)
        {
            var start = GetInt(citation, "start_block_index");
            var end = GetInt(citation, "end_block_index");
            return start != null && end != null
                ? new ContentBlockRangeLocation
                {
                    StartBlockIndex = start.Value,
                    EndBlockIndex = end.Value,
                    EndExclusive = true
                }
                : null;
        }

        private static void AddIntMetadata(
            Dictionary<string, object> metadata,
            JsonElement element,
            string property,
            string key)
        {
            if (GetInt(element, property) is int value)
                metadata[key] = value;
        }

        private static void AddStringMetadata(
            Dictionary<string, object> metadata,
            JsonElement element,
            string property,
            string key)
        {
            if (GetString(element, property) is string value)
                metadata[key] = value;
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

        private static bool TryGetClaudeNativeValue(ContentBlock block, out JsonElement value)
        {
            if (HasClaudeNativeRepresentation(block))
            {
                value = block.NativeRepresentation!.Value.Clone();
                return true;
            }

            value = default;
            return false;
        }

        private static ClaudeOutputConfig? ConvertOutputConfig(
            OutputFormat? output,
            ReasoningOptions? reasoning)
        {
            var effort = reasoning?.Effort == null
                ? null
                : ToWireValue(reasoning.Effort.Value);
            if ((output == null || output.Kind == OutputFormatKind.Text) && effort == null)
                return null;

            var outputConfig = new ClaudeOutputConfig
            {
                Effort = effort
            };

            switch (output?.Kind)
            {
                case OutputFormatKind.JsonObject:
                    outputConfig.Format = new ClaudeOutputFormat
                    {
                        Type = "json_schema",
                        Schema = new Dictionary<string, object>
                        {
                            { "type", "object" },
                            { "additionalProperties", true }
                        }
                    };
                    break;

                case OutputFormatKind.JsonSchema:
                    if (output.JsonSchema != null)
                    {
                        outputConfig.Format = new ClaudeOutputFormat
                        {
                            Type = "json_schema",
                            Schema = output.JsonSchema.Schema
                        };
                    }
                    break;
            }

            return outputConfig;
        }

        private static ClaudeThinkingConfig? ConvertReasoning(ReasoningOptions? reasoning)
        {
            if (reasoning == null)
                return null;

            var options = reasoning.Anthropic;
            if (options == null || options.Mode == AnthropicThinkingMode.Default)
                return null;

            return new ClaudeThinkingConfig
            {
                Type = options.Mode switch
                {
                    AnthropicThinkingMode.Disabled => "disabled",
                    AnthropicThinkingMode.Adaptive => "adaptive",
                    AnthropicThinkingMode.Manual => "enabled",
                    _ => throw new ArgumentOutOfRangeException()
                },
                BudgetTokens = options.Mode == AnthropicThinkingMode.Manual
                    ? options.BudgetTokens
                    : null,
                Display = options.Mode == AnthropicThinkingMode.Disabled
                    ? null
                    : reasoning.Output switch
                    {
                        ReasoningOutput.Omitted => "omitted",
                        ReasoningOutput.Summary => "summarized",
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

        private FinishReason ConvertFinishReason(string? reason)
        {
            return reason switch
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
        }
    }
}
