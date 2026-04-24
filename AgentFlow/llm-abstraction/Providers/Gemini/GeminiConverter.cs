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
                    MaxOutputTokens = request.Parameters.MaxTokens,
                    Temperature = request.Parameters.Temperature,
                    TopP = request.Parameters.TopP,
                    TopK = request.Parameters.TopK,
                    StopSequences = request.Parameters.StopSequences
                }
            };

            // Add system instruction if present
            if (!string.IsNullOrEmpty(request.System))
            {
                geminiRequest.SystemInstruction = new GeminiContent
                {
                    Parts = new List<GeminiPart>
                    {
                        new GeminiPart { Text = request.System }
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
                Id = Guid.NewGuid().ToString(), // Gemini doesn't provide ID
                Model = string.Empty, // Model info not in response
                Choices = choices,
                Usage = new UsageInfo
                {
                    PromptTokens = response.UsageMetadata?.PromptTokenCount ?? 0,
                    CompletionTokens = response.UsageMetadata?.CandidatesTokenCount ?? 0,
                    TotalTokens = response.UsageMetadata?.TotalTokenCount ?? 0
                }
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

                    case ToolCallContent toolCall:
                        parts.Add(new GeminiPart
                        {
                            FunctionCall = new GeminiFunctionCall
                            {
                                Name = toolCall.Name,
                                Args = toolCall.Input
                            }
                        });
                        break;

                    case ToolResultContent toolResult:
                        parts.Add(new GeminiPart
                        {
                            FunctionResponse = new GeminiFunctionResponse
                            {
                                Name = toolResult.ToolCallId, // Use ID as name
                                Response = new Dictionary<string, object>
                                {
                                    { "result", toolResult.Output }
                                }
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
                ToolChoiceType.Specific => "ANY", // Gemini doesn't support specific tool selection
                _ => "AUTO"
            };

            return new GeminiToolConfig
            {
                FunctionCallingConfig = new GeminiFunctionCallingConfig
                {
                    Mode = mode
                }
            };
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
                        message.Content.Add(new ToolCallContent
                        {
                            Id = Guid.NewGuid().ToString(), // Generate ID
                            Name = part.FunctionCall.Name,
                            Input = part.FunctionCall.Args ?? new Dictionary<string, object>()
                        });
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
                        request.GenerationConfig.ResponseSchema = responseFormat.JsonSchema.Schema;
                    }
                    break;
            }
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
