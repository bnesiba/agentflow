using System;
using System.Collections.Generic;
using System.Linq;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.OpenAI.Models;

namespace LLMAbstraction.Providers.OpenAI
{
    /// <summary>
    /// Converts between unified models and OpenAI-specific models
    /// </summary>
    public class OpenAIConverter : IModelConverter<OpenAIChatRequest, OpenAIChatResponse>
    {
        public OpenAIChatRequest ConvertRequest(UnifiedRequest request)
        {
            var openAIRequest = new OpenAIChatRequest
            {
                Model = request.Model,
                Messages = ConvertMessages(request.Messages, request.System),
                MaxTokens = request.Parameters.MaxTokens,
                Temperature = request.Parameters.Temperature,
                TopP = request.Parameters.TopP,
                Stop = request.Parameters.StopSequences,
                Stream = request.Parameters.Stream
            };

            // Convert tools if present
            if (request.Tools != null && request.Tools.Any())
            {
                openAIRequest.Tools = request.Tools.Select(ConvertTool).ToList();
                openAIRequest.ToolChoice = ConvertToolChoice(request.ToolChoice);
            }

            // Convert response format if present
            if (request.ResponseFormat != null)
            {
                openAIRequest.ResponseFormat = ConvertResponseFormat(request.ResponseFormat);
            }

            return openAIRequest;
        }

        public UnifiedResponse ConvertResponse(OpenAIChatResponse response)
        {
            return new UnifiedResponse
            {
                Id = response.Id,
                Model = response.Model,
                Choices = response.Choices.Select(ConvertChoice).ToList(),
                Usage = new UsageInfo
                {
                    PromptTokens = response.Usage.PromptTokens,
                    CompletionTokens = response.Usage.CompletionTokens,
                    TotalTokens = response.Usage.TotalTokens
                }
            };
        }

        private List<OpenAIMessage> ConvertMessages(List<UnifiedMessage> messages, string? system)
        {
            var result = new List<OpenAIMessage>();

            // Add system message if present
            if (!string.IsNullOrEmpty(system))
            {
                result.Add(new OpenAIMessage
                {
                    Role = "system",
                    Content = system
                });
            }

            // Convert all messages
            foreach (var message in messages)
            {
                result.Add(ConvertMessage(message));
            }

            return result;
        }

        private OpenAIMessage ConvertMessage(UnifiedMessage message)
        {
            var role = message.Role switch
            {
                MessageRole.System => "system",
                MessageRole.User => "user",
                MessageRole.Assistant => "assistant",
                MessageRole.Tool => "tool",
                _ => throw new ArgumentException($"Unknown role: {message.Role}")
            };

            // Check if this is a simple text message
            if (message.Content.Count == 1 && message.Content[0] is TextContent textContent)
            {
                return new OpenAIMessage
                {
                    Role = role,
                    Content = textContent.Text
                };
            }

            // Handle multimodal or tool content
            var contentParts = new List<object>();
            List<OpenAIToolCall>? toolCalls = null;
            string? toolCallId = null;

            foreach (var block in message.Content)
            {
                switch (block)
                {
                    case TextContent text:
                        contentParts.Add(new { type = "text", text = text.Text });
                        break;

                    case ImageContent image:
                        if (!string.IsNullOrEmpty(image.Source.Url))
                        {
                            contentParts.Add(new
                            {
                                type = "image_url",
                                image_url = new { url = image.Source.Url }
                            });
                        }
                        else if (!string.IsNullOrEmpty(image.Source.Data))
                        {
                            contentParts.Add(new
                            {
                                type = "image_url",
                                image_url = new
                                {
                                    url = $"data:{image.Source.MediaType};base64,{image.Source.Data}"
                                }
                            });
                        }
                        break;

                    case ToolCallContent toolCall:
                        toolCalls ??= new List<OpenAIToolCall>();
                        toolCalls.Add(new OpenAIToolCall
                        {
                            Id = toolCall.Id,
                            Type = "function",
                            Function = new OpenAIFunctionCall
                            {
                                Name = toolCall.Name,
                                Arguments = System.Text.Json.JsonSerializer.Serialize(toolCall.Input)
                            }
                        });
                        break;

                    case ToolResultContent toolResult:
                        toolCallId = toolResult.ToolCallId;
                        contentParts.Add(new { type = "text", text = toolResult.Output });
                        break;
                }
            }

            var openAIMessage = new OpenAIMessage { Role = role };

            if (toolCalls != null)
            {
                openAIMessage.ToolCalls = toolCalls;
            }

            if (toolCallId != null)
            {
                openAIMessage.ToolCallId = toolCallId;
            }

            if (contentParts.Any())
            {
                openAIMessage.Content = contentParts.Count == 1 && contentParts[0] is string str
                    ? str
                    : contentParts;
            }

            return openAIMessage;
        }

        private OpenAITool ConvertTool(ToolDefinition tool)
        {
            return new OpenAITool
            {
                Type = "function",
                Function = new OpenAIFunction
                {
                    Name = tool.Name,
                    Description = tool.Description,
                    Parameters = tool.Parameters
                }
            };
        }

        private object? ConvertToolChoice(ToolChoice? toolChoice)
        {
            if (toolChoice == null)
                return null;

            return toolChoice.Type switch
            {
                ToolChoiceType.Auto => "auto",
                ToolChoiceType.None => "none",
                ToolChoiceType.Required => "required",
                ToolChoiceType.Specific => new
                {
                    type = "function",
                    function = new { name = toolChoice.ToolName }
                },
                _ => "auto"
            };
        }

        private object? ConvertResponseFormat(ResponseFormat responseFormat)
        {
            return responseFormat.Type switch
            {
                ResponseFormatType.Text => new { type = "text" },
                ResponseFormatType.Json => new { type = "json_object" },
                ResponseFormatType.JsonSchema => new
                {
                    type = "json_schema",
                    json_schema = new
                    {
                        name = responseFormat.JsonSchema!.Name,
                        description = responseFormat.JsonSchema.Description,
                        schema = responseFormat.JsonSchema.Schema,
                        strict = responseFormat.JsonSchema.Strict
                    }
                },
                _ => null
            };
        }

        private ResponseChoice ConvertChoice(OpenAIChoice choice)
        {
            var message = new UnifiedMessage
            {
                Role = MessageRole.Assistant,
                Content = new List<ContentBlock>()
            };

            // Handle text content
            if (choice.Message.Content is string textContent && !string.IsNullOrEmpty(textContent))
            {
                message.Content.Add(new TextContent { Text = textContent });
            }

            // Handle tool calls
            if (choice.Message.ToolCalls != null)
            {
                foreach (var toolCall in choice.Message.ToolCalls)
                {
                    var input = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(
                        toolCall.Function.Arguments) ?? new Dictionary<string, object>();

                    message.Content.Add(new ToolCallContent
                    {
                        Id = toolCall.Id,
                        Name = toolCall.Function.Name,
                        Input = input
                    });
                }
            }

            return new ResponseChoice
            {
                Index = choice.Index,
                Message = message,
                FinishReason = ConvertFinishReason(choice.FinishReason)
            };
        }

        private FinishReason ConvertFinishReason(string? reason)
        {
            return reason switch
            {
                "stop" => FinishReason.Stop,
                "length" => FinishReason.MaxTokens,
                "tool_calls" => FinishReason.ToolCalls,
                "content_filter" => FinishReason.ContentFilter,
                "function_call" => FinishReason.ToolCalls,
                _ => FinishReason.Other
            };
        }
    }
}
