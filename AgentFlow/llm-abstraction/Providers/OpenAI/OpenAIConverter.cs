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
            "text", "reasoning", "user", "metadata"
        };
        private static readonly JsonSerializerOptions NativeJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        public OpenAIResponseRequest ConvertRequest(UnifiedRequest request)
        {
            LLMRequestValidator.ValidateAndThrow(request, LLMProvider.OpenAI);
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
                AdditionalProperties = ProviderOptionMerger.ConvertAdditionalFields(
                    request.ProviderOptions?.OpenAI,
                    ProtectedRequestFields)
            };

            if (request.Continuation?.Provider == ProviderIds.OpenAI &&
                request.Continuation.Mode == ContinuationMode.ServerManaged)
            {
                openAIRequest.PreviousResponseId = request.Continuation.ResponseId;
            }

            if (request.Tools != null && request.Tools.Any())
            {
                openAIRequest.Tools = request.Tools.Select(ConvertTool).ToList();
                openAIRequest.ToolChoice = ConvertToolChoice(request.ToolChoice);

                if (request.ToolChoice?.DisableParallelToolUse != null)
                {
                    openAIRequest.ParallelToolCalls = !request.ToolChoice.DisableParallelToolUse.Value;
                }
            }

            if (request.ResponseFormat != null)
            {
                openAIRequest.Text = new OpenAITextConfig
                {
                    Format = ConvertResponseFormat(request.ResponseFormat)
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
                            Input = DeserializeArguments(item.Arguments)
                        });
                        break;
                }
            }

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
                            messageContent.Add(new Dictionary<string, object?>
                            {
                                { "type", "input_text" },
                                { "text", text.Text }
                            });
                            break;

                        case ImageContent image:
                            AddImageContent(messageContent, image.Source);
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

        private static void AddImageContent(List<object> content, ImageSource source)
        {
            if (!string.IsNullOrEmpty(source.Url))
            {
                content.Add(new Dictionary<string, object?>
                {
                    { "type", "input_image" },
                    { "detail", "auto" },
                    { "image_url", source.Url }
                });
            }
            else if (!string.IsNullOrEmpty(source.Data))
            {
                content.Add(new Dictionary<string, object?>
                {
                    { "type", "input_image" },
                    { "detail", "auto" },
                    { "image_url", $"data:{source.MediaType ?? "image/jpeg"};base64,{source.Data}" }
                });
            }
            else
            {
                throw new NotSupportedException("OpenAI image content requires a URL or base64 data.");
            }
        }

        private static void AddMediaContent(List<object> content, MediaContent media)
        {
            if (media.MediaType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
            {
                if (!string.IsNullOrEmpty(media.Source.FileId))
                {
                    content.Add(new Dictionary<string, object?>
                    {
                        { "type", "input_image" },
                        { "detail", "auto" },
                        { "file_id", media.Source.FileId }
                    });
                }
                else if (!string.IsNullOrEmpty(media.Source.Url))
                {
                    content.Add(new Dictionary<string, object?>
                    {
                        { "type", "input_image" },
                        { "detail", "auto" },
                        { "image_url", media.Source.Url }
                    });
                }
                else if (!string.IsNullOrEmpty(media.Source.Base64Data))
                {
                    content.Add(new Dictionary<string, object?>
                    {
                        { "type", "input_image" },
                        { "detail", "auto" },
                        { "image_url", $"data:{media.MediaType};base64,{media.Source.Base64Data}" }
                    });
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
                content.Add(new Dictionary<string, object?>
                {
                    { "type", "input_file" },
                    { "file_id", media.Source.FileId }
                });
            }
            else if (!string.IsNullOrEmpty(media.Source.Url))
            {
                content.Add(new Dictionary<string, object?>
                {
                    { "type", "input_file" },
                    { "file_url", media.Source.Url }
                });
            }
            else if (!string.IsNullOrEmpty(media.Source.Base64Data))
            {
                content.Add(new Dictionary<string, object?>
                {
                    { "type", "input_file" },
                    { "filename", media.Source.FileName ?? "input" },
                    { "file_data", $"data:{media.MediaType};base64,{media.Source.Base64Data}" }
                });
            }
            else
            {
                throw new NotSupportedException(
                    "OpenAI media content requires a file ID, URL, or base64 data. Provider file URIs are not accepted as file URLs.");
            }
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

        private static OpenAIResponseTool ConvertTool(ToolDefinition tool)
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

        private static object? ConvertToolChoice(ToolChoice? toolChoice)
        {
            if (toolChoice == null)
                return null;

            return toolChoice.Type switch
            {
                ToolChoiceType.Auto => "auto",
                ToolChoiceType.None => "none",
                ToolChoiceType.Required => "required",
                ToolChoiceType.Specific => new Dictionary<string, object?>
                {
                    { "type", "function" },
                    { "name", toolChoice.ToolName }
                },
                _ => "auto"
            };
        }

        private static object? ConvertResponseFormat(ResponseFormat responseFormat)
        {
            return responseFormat.Type switch
            {
                ResponseFormatType.Text => new Dictionary<string, object?>
                {
                    { "type", "text" }
                },
                ResponseFormatType.Json => new Dictionary<string, object?>
                {
                    { "type", "json_object" }
                },
                ResponseFormatType.JsonSchema => new Dictionary<string, object?>
                {
                    { "type", "json_schema" },
                    { "name", responseFormat.JsonSchema!.Name },
                    { "description", responseFormat.JsonSchema.Description },
                    { "schema", responseFormat.JsonSchema.Schema },
                    { "strict", responseFormat.JsonSchema.Strict }
                },
                _ => null
            };
        }

        private static OpenAIReasoningConfig? ConvertReasoning(ReasoningOptions reasoning)
        {
            if (reasoning.Enabled == false && string.IsNullOrEmpty(reasoning.Effort))
            {
                return new OpenAIReasoningConfig { Effort = "none" };
            }

            if (string.IsNullOrEmpty(reasoning.Effort) && string.IsNullOrEmpty(reasoning.Summary))
                return null;

            return new OpenAIReasoningConfig
            {
                Effort = reasoning.Effort,
                Summary = reasoning.Summary ?? (reasoning.IncludeThoughts == true ? "auto" : null)
            };
        }

        private static void AddMessageContent(UnifiedMessage message, OpenAIOutputItem item)
        {
            if (item.Content == null)
                return;

            foreach (var content in item.Content)
            {
                if ((content.Type == "output_text" || content.Type == "text") && content.Text != null)
                {
                    var text = new TextContent { Text = content.Text };
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
