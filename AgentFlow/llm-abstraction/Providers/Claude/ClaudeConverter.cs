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
            "stop_sequences", "stream", "tools", "tool_choice", "output_config", "thinking", "metadata"
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
            LLMRequestValidator.ValidateAndThrow(request, LLMProvider.Claude);
            var thinking = ConvertReasoning(request.Reasoning);

            var claudeRequest = new ClaudeMessageRequest
            {
                Model = request.Model,
                MaxTokens = request.Parameters.MaxOutputTokens ?? 1024, // Claude requires max_tokens
                Messages = ConvertMessages(request.Messages),
                System = UnifiedRequestNormalization.CombineInstructions(request),
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
            if (request.Tools != null && request.Tools.Any())
            {
                claudeRequest.Tools = request.Tools.Select(ConvertTool).ToList();
                claudeRequest.ToolChoice = ConvertToolChoice(request.ToolChoice);
            }

            // Convert response format if present (Claude native support via output_config)
            if (request.ResponseFormat != null)
            {
                claudeRequest.OutputConfig = ConvertResponseFormat(request.ResponseFormat);
            }

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
            return new UnifiedResponse
            {
                Id = response.Id,
                Model = response.Model,
                Choices = new List<ResponseChoice>
                {
                    new ResponseChoice
                    {
                        Index = 0,
                        Message = ConvertMessage(response),
                        FinishReason = ConvertFinishReason(response.StopReason)
                    }
                },
                Usage = new UsageInfo
                {
                    InputTokens = response.Usage.InputTokens,
                    OutputTokens = response.Usage.OutputTokens,
                    TotalTokens = response.Usage.InputTokens + response.Usage.OutputTokens,
                    CacheCreationTokens = response.Usage.CacheCreationInputTokens,
                    CacheReadTokens = response.Usage.CacheReadInputTokens
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
                        blocks.Add(new
                        {
                            type = "text",
                            text = text.Text
                        });
                        break;

                    case ImageContent image:
                        if (!string.IsNullOrEmpty(image.Source.Url))
                        {
                            blocks.Add(new
                            {
                                type = "image",
                                source = new
                                {
                                    type = "url",
                                    url = image.Source.Url
                                }
                            });
                        }
                        else if (!string.IsNullOrEmpty(image.Source.Data))
                        {
                            blocks.Add(new
                            {
                                type = "image",
                                source = new
                                {
                                    type = "base64",
                                    media_type = image.Source.MediaType,
                                    data = image.Source.Data
                                }
                            });
                        }
                        else
                        {
                            throw new NotSupportedException("Claude image content requires a URL or base64 data.");
                        }
                        break;

                    case MediaContent media:
                        blocks.Add(ConvertMediaContent(media));
                        break;

                    case ToolCallContent toolCall:
                        blocks.Add(new
                        {
                            type = "tool_use",
                            id = toolCall.Id,
                            name = toolCall.Name,
                            input = toolCall.Input
                        });
                        break;

                    case ToolResultContent toolResult:
                        blocks.Add(new
                        {
                            type = "tool_result",
                            tool_use_id = toolResult.ToolCallId,
                            content = toolResult.Output,
                            is_error = toolResult.IsError
                        });
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

        private static object ConvertMediaContent(MediaContent media)
        {
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

        private ClaudeTool ConvertTool(ToolDefinition tool)
        {
            return new ClaudeTool
            {
                Name = tool.Name,
                Description = tool.Description,
                InputSchema = tool.Parameters,
                Strict = tool.Strict
            };
        }

        private object? ConvertToolChoice(ToolChoice? toolChoice)
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
                    name = toolChoice.ToolName
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
                return new TextContent
                {
                    Text = block.Text ?? string.Empty,
                    NativeRepresentation = nativeRepresentation
                };
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

        private ClaudeOutputConfig? ConvertResponseFormat(ResponseFormat responseFormat)
        {
            if (responseFormat.Type == ResponseFormatType.Text)
                return null;

            var outputConfig = new ClaudeOutputConfig
            {
                Format = new ClaudeOutputFormat()
            };

            switch (responseFormat.Type)
            {
                case ResponseFormatType.Json:
                    outputConfig.Format.Type = "json_schema";
                    outputConfig.Format.Schema = new Dictionary<string, object>
                    {
                        { "type", "object" },
                        { "additionalProperties", true }
                    };
                    break;

                case ResponseFormatType.JsonSchema:
                    outputConfig.Format.Type = "json_schema";
                    if (responseFormat.JsonSchema != null)
                    {
                        outputConfig.Format.Schema = responseFormat.JsonSchema.Schema;
                    }
                    break;
            }

            return outputConfig;
        }

        private static ClaudeThinkingConfig? ConvertReasoning(ReasoningOptions? reasoning)
        {
            if (reasoning == null)
                return null;

            if (reasoning.Enabled == false)
            {
                return new ClaudeThinkingConfig { Type = "disabled" };
            }

            if (reasoning.BudgetTokens != null)
            {
                return new ClaudeThinkingConfig
                {
                    Type = "enabled",
                    BudgetTokens = reasoning.BudgetTokens
                };
            }

            if (!string.IsNullOrEmpty(reasoning.Effort) &&
                reasoning.Effort.Equals("adaptive", StringComparison.OrdinalIgnoreCase))
            {
                return new ClaudeThinkingConfig { Type = "adaptive" };
            }

            return null;
        }

        private FinishReason ConvertFinishReason(string? reason)
        {
            return reason switch
            {
                "end_turn" => FinishReason.Stop,
                "max_tokens" => FinishReason.MaxTokens,
                "stop_sequence" => FinishReason.Stop,
                "tool_use" => FinishReason.ToolCalls,
                "refusal" => FinishReason.ContentFilter,
                "model_context_window_exceeded" => FinishReason.MaxTokens,
                _ => FinishReason.Other
            };
        }
    }
}
