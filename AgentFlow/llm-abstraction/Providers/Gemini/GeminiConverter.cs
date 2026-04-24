using System;
using System.Collections.Generic;
using System.Linq;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini.Models;

namespace LLMAbstraction.Providers.Gemini
{
    /// <summary>
    /// Converts between unified models and Gemini-specific models
    /// </summary>
    public class GeminiConverter : IModelConverter<GeminiGenerateRequest, GeminiGenerateResponse>
    {
        public GeminiGenerateRequest ConvertRequest(UnifiedRequest request)
        {
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
                }
            };

            // Add system instruction if present
            if (!string.IsNullOrEmpty(request.Instructions))
            {
                geminiRequest.SystemInstruction = new GeminiContent
                {
                    Parts = new List<GeminiPart>
                    {
                        new GeminiPart { Text = request.Instructions }
                    }
                };
            }

            // Convert tools if present
            if (request.Tools != null && request.Tools.Any())
            {
                geminiRequest.Tools = new List<GeminiTool>
                {
                    new GeminiTool
                    {
                        FunctionDeclarations = request.Tools.Select(ConvertTool).ToList()
                    }
                };

                geminiRequest.ToolConfig = ConvertToolChoice(request.ToolChoice);
            }

            // Convert response format if present
            if (request.ResponseFormat != null)
            {
                ConvertResponseFormat(geminiRequest, request.ResponseFormat);
            }

            if (request.ProviderOptions?.Gemini != null &&
                request.ProviderOptions.Gemini.TryGetValue("safetySettings", out var safetySettings) &&
                safetySettings is List<Dictionary<string, object>> typedSafetySettings)
            {
                geminiRequest.SafetySettings = typedSafetySettings;
            }

            return geminiRequest;
        }

        public UnifiedResponse ConvertResponse(GeminiGenerateResponse response)
        {
            var choices = new List<ResponseChoice>();

            if (response.Candidates != null)
            {
                for (int i = 0; i < response.Candidates.Count; i++)
                {
                    var candidate = response.Candidates[i];
                    choices.Add(new ResponseChoice
                    {
                        Index = i,
                        Message = ConvertContent(candidate.Content),
                        FinishReason = ConvertFinishReason(candidate.FinishReason)
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
                    ReasoningTokens = response.UsageMetadata?.ThoughtsTokenCount
                },
                ProviderMetadata = response.PromptFeedback != null
                    ? new Dictionary<string, object> { { "promptFeedback", response.PromptFeedback } }
                    : null
            };
        }

        private List<GeminiContent> ConvertMessages(List<UnifiedMessage> messages)
        {
            var result = new List<GeminiContent>();

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

                var parts = ConvertContentBlocks(message.Content);

                result.Add(new GeminiContent
                {
                    Role = role,
                    Parts = parts
                });
            }

            return result;
        }

        private List<GeminiPart> ConvertContentBlocks(List<ContentBlock> content)
        {
            var parts = new List<GeminiPart>();

            foreach (var block in content)
            {
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
                        // Note: Gemini doesn't support URL-based images directly
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
                        parts.Add(new GeminiPart
                        {
                            FunctionResponse = new GeminiFunctionResponse
                            {
                                Name = toolResult.ToolName ?? toolResult.ToolCallId,
                                Id = toolResult.ToolCallId,
                                Response = ConvertToolResultOutput(toolResult.Output)
                            }
                        });
                        break;
                }
            }

            return parts;
        }

        private GeminiFunctionDeclaration ConvertTool(ToolDefinition tool)
        {
            return new GeminiFunctionDeclaration
            {
                Name = tool.Name,
                Description = tool.Description,
                Parameters = tool.Parameters
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
                    if (!string.IsNullOrEmpty(part.Text))
                    {
                        message.Content.Add(new TextContent { Text = part.Text });
                    }
                    else if (part.FunctionCall != null)
                    {
                        var toolCall = new ToolCallContent
                        {
                            Id = part.FunctionCall.Id ?? Guid.NewGuid().ToString(),
                            Name = part.FunctionCall.Name,
                            Input = part.FunctionCall.Args ?? new Dictionary<string, object>()
                        };
                        if (!string.IsNullOrEmpty(part.ThoughtSignature))
                        {
                            toolCall.ProviderMetadata = new Dictionary<string, object>
                            {
                                { "gemini.thoughtSignature", part.ThoughtSignature }
                            };
                        }

                        message.Content.Add(toolCall);
                    }
                }
            }

            return message;
        }

        private void ConvertResponseFormat(GeminiGenerateRequest request, ResponseFormat responseFormat)
        {
            // Gemini uses responseMimeType and responseSchema in GenerationConfig
            if (request.GenerationConfig == null)
                request.GenerationConfig = new GeminiGenerationConfig();

            switch (responseFormat.Type)
            {
                case ResponseFormatType.Json:
                    request.GenerationConfig.ResponseMimeType = "application/json";
                    break;

                case ResponseFormatType.JsonSchema:
                    request.GenerationConfig.ResponseMimeType = "application/json";
                    if (responseFormat.JsonSchema != null)
                    {
                        request.GenerationConfig.ResponseJsonSchema = responseFormat.JsonSchema.Schema;
                    }
                    break;
            }
        }

        private static GeminiThinkingConfig? ConvertReasoning(ReasoningOptions? reasoning)
        {
            if (reasoning == null)
                return null;

            var config = new GeminiThinkingConfig
            {
                IncludeThoughts = reasoning.IncludeThoughts,
                ThinkingBudget = reasoning.Enabled == false ? 0 : reasoning.BudgetTokens
            };

            if (!string.IsNullOrEmpty(reasoning.Effort))
            {
                config.ThinkingLevel = reasoning.Effort.ToUpperInvariant();
            }

            return config;
        }

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

            var fileUri = media.Source.FileUri ?? media.Source.Url;
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
            }
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
                _ => FinishReason.Other
            };
        }
    }
}
