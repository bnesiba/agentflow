using System.Text.Json;
using System.Text.Json.Nodes;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class ProviderRequestContractTests
{
    [Fact]
    public void ClaudeAdvancedRequestMatchesMessagesApiShape()
    {
        var request = new UnifiedRequest
        {
            Model = "claude-sonnet-4-6",
            Instructions = "Be concise.",
            Messages = { new UnifiedMessage(MessageRole.User, "Weather?") },
            Parameters = new GenerationParameters
            {
                MaxOutputTokens = 4096,
                TopP = 0.95,
                StopSequences = new List<string> { "END" }
            },
            Reasoning = new ReasoningOptions
            {
                Effort = ReasoningEffort.Medium,
                Output = ReasoningOutput.Summary,
                Anthropic = new AnthropicReasoningOptions
                {
                    Mode = AnthropicThinkingMode.Manual,
                    BudgetTokens = 1024
                }
            },
            Tools = new List<ToolDefinition> { WeatherTool(strict: true) },
            ToolChoice = new ToolChoice { Type = ToolChoiceType.Auto },
            Output = WeatherOutputFormat()
        };

        var json = Serialize(new ClaudeConverter().ConvertRequest(request));

        Assert.Equal("claude-sonnet-4-6", Text(json, "model"));
        Assert.Equal(4096, Number(json, "max_tokens"));
        Assert.Equal("Be concise.", Text(json, "system"));
        Assert.Null(json["temperature"]); // incompatible with manual thinking
        Assert.Equal(0.95, Double(json, "top_p"));
        Assert.Null(json["top_k"]); // incompatible with manual thinking
        Assert.Equal("enabled", Text(json["thinking"]!, "type"));
        Assert.Equal(1024, Number(json["thinking"]!, "budget_tokens"));
        Assert.Equal("summarized", Text(json["thinking"]!, "display"));
        Assert.Equal("medium", Text(json["output_config"]!, "effort"));
        Assert.Equal("get_weather", Text(json["tools"]![0]!, "name"));
        Assert.Equal("object", Text(json["tools"]![0]!["input_schema"]!, "type"));
        Assert.Equal("auto", Text(json["tool_choice"]!, "type"));
        Assert.Equal("json_schema", Text(json["output_config"]!["format"]!, "type"));
    }

    [Fact]
    public void ClaudePlainTextResponseFormatDoesNotEmitInvalidEmptyFormat()
    {
        var request = BasicRequest("claude-sonnet-4-6");
        request.Output = new OutputFormat { Kind = OutputFormatKind.Text };

        var json = Serialize(new ClaudeConverter().ConvertRequest(request));

        Assert.Null(json["output_config"]);
    }

    [Fact]
    public void OpenAIAdvancedRequestMatchesResponsesApiShape()
    {
        var request = BasicRequest("gpt-5.6");
        request.Instructions = "Be concise.";
        request.Messages[0].Content.Add(new ImageContent
        {
            Source = new ImageSource { Url = "https://example.com/map.png" }
        });
        request.Messages[0].Content.Add(new MediaContent
        {
            MediaType = "application/pdf",
            Source = new MediaSource { FileId = "file_123" }
        });
        request.Tools = new List<ToolDefinition> { WeatherTool(strict: true) };
        request.ToolChoice = new ToolChoice
        {
            Type = ToolChoiceType.Specific,
            ToolName = "get_weather",
            DisableParallelToolUse = true
        };
        request.Output = WeatherOutputFormat();
        request.Reasoning = new ReasoningOptions
        {
            Effort = ReasoningEffort.Medium,
            Output = ReasoningOutput.Summary,
            OpenAI = new OpenAIReasoningOptions
            {
                Mode = OpenAIReasoningMode.Pro,
                Context = OpenAIReasoningContext.AllTurns,
                Summary = OpenAIReasoningSummary.Auto
            }
        };

        var json = Serialize(new OpenAIConverter().ConvertRequest(request));

        Assert.Equal("input_text", Text(json["input"]![0]!["content"]![0]!, "type"));
        Assert.Equal("input_image", Text(json["input"]![0]!["content"]![1]!, "type"));
        Assert.Equal("input_file", Text(json["input"]![0]!["content"]![2]!, "type"));
        Assert.Equal("file_123", Text(json["input"]![0]!["content"]![2]!, "file_id"));
        Assert.Equal("function", Text(json["tools"]![0]!, "type"));
        Assert.True(json["tools"]![0]!["strict"]!.GetValue<bool>());
        Assert.False(json["parallel_tool_calls"]!.GetValue<bool>());
        Assert.Equal("function", Text(json["tool_choice"]!, "type"));
        Assert.Equal("json_schema", Text(json["text"]!["format"]!, "type"));
        Assert.Equal("medium", Text(json["reasoning"]!, "effort"));
        Assert.Equal("pro", Text(json["reasoning"]!, "mode"));
        Assert.Equal("all_turns", Text(json["reasoning"]!, "context"));
        Assert.Equal("auto", Text(json["reasoning"]!, "summary"));
    }

    [Fact]
    public void GeminiAdvancedRequestMatchesGenerateContentShape()
    {
        var request = BasicRequest("gemini-3.5-flash");
        request.Instructions = "Be concise.";
        request.Messages[0].Content.Add(new MediaContent
        {
            MediaType = "application/pdf",
            Source = new MediaSource
            {
                FileUri = "https://generativelanguage.googleapis.com/v1beta/files/abc",
                FileName = "report.pdf"
            }
        });
        request.Tools = new List<ToolDefinition> { WeatherTool(strict: null) };
        request.ToolChoice = new ToolChoice
        {
            Type = ToolChoiceType.Specific,
            ToolName = "get_weather"
        };
        request.Output = WeatherOutputFormat();
        request.Reasoning = new ReasoningOptions
        {
            Effort = ReasoningEffort.Medium,
            Output = ReasoningOutput.Summary
        };

        var json = Serialize(new GeminiConverter().ConvertRequest(request));

        Assert.Equal("Be concise.", Text(json["systemInstruction"]!["parts"]![0]!, "text"));
        Assert.Equal("user", Text(json["contents"]![0]!, "role"));
        Assert.Equal("application/pdf", Text(json["contents"]![0]!["parts"]![1]!["fileData"]!, "mimeType"));
        Assert.Equal("get_weather", Text(json["tools"]![0]!["functionDeclarations"]![0]!, "name"));
        Assert.Equal("ANY", Text(json["toolConfig"]!["functionCallingConfig"]!, "mode"));
        Assert.Equal("get_weather",
            json["toolConfig"]!["functionCallingConfig"]!["allowedFunctionNames"]![0]!.GetValue<string>());
        Assert.Equal("application/json", Text(json["generationConfig"]!, "responseMimeType"));
        Assert.Equal("MEDIUM", Text(json["generationConfig"]!["thinkingConfig"]!, "thinkingLevel"));
        Assert.True(json["generationConfig"]!["thinkingConfig"]!["includeThoughts"]!.GetValue<bool>());
    }

    [Fact]
    public void ToolResultWireShapesUseEachProvidersContinuationIdentifier()
    {
        var assistant = new UnifiedMessage(MessageRole.Assistant, new List<ContentBlock>
        {
            new ToolCallContent { Id = "call_1", Name = "get_weather" }
        });
        var result = new UnifiedMessage(MessageRole.Tool, new List<ContentBlock>
        {
            new ToolResultContent { ToolCallId = "call_1", ToolName = "get_weather", Output = "72 F" }
        });

        var claude = BasicRequest("claude-sonnet-4-6");
        claude.Messages = new List<UnifiedMessage> { assistant, result };
        var claudeJson = Serialize(new ClaudeConverter().ConvertRequest(claude));
        Assert.Equal("call_1", Text(claudeJson["messages"]![1]!["content"]![0]!, "tool_use_id"));

        var openAI = BasicRequest("gpt-5.6");
        openAI.Messages = new List<UnifiedMessage> { assistant, result };
        var openAIJson = Serialize(new OpenAIConverter().ConvertRequest(openAI));
        Assert.Equal("call_1", Text(openAIJson["input"]![1]!, "call_id"));

        var gemini = BasicRequest("gemini-3.5-flash");
        gemini.Messages = new List<UnifiedMessage> { assistant, result };
        var geminiJson = Serialize(new GeminiConverter().ConvertRequest(gemini));
        Assert.Equal("call_1", Text(geminiJson["contents"]![1]!["parts"]![0]!["functionResponse"]!, "id"));
        Assert.Equal("get_weather", Text(geminiJson["contents"]![1]!["parts"]![0]!["functionResponse"]!, "name"));
    }

    private static UnifiedRequest BasicRequest(string model) => new()
    {
        Model = model,
        Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
    };

    private static ToolDefinition WeatherTool(bool? strict) => new()
    {
        Name = "get_weather",
        Description = "Get weather.",
        Strict = strict,
        Parameters = new Dictionary<string, object>
        {
            ["type"] = "object",
            ["properties"] = new Dictionary<string, object>
            {
                ["city"] = new Dictionary<string, object> { ["type"] = "string" }
            },
            ["required"] = new[] { "city" },
            ["additionalProperties"] = false
        }
    };

    private static OutputFormat WeatherOutputFormat() => new()
    {
        Kind = OutputFormatKind.JsonSchema,
        JsonSchema = new JsonSchemaDefinition
        {
            Name = "weather",
            Schema = new Dictionary<string, object>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object>
                {
                    ["temperature"] = new Dictionary<string, object> { ["type"] = "number" }
                },
                ["required"] = new[] { "temperature" },
                ["additionalProperties"] = false
            }
        }
    };

    private static JsonObject Serialize(object value)
        => JsonSerializer.SerializeToNode(value, new JsonSerializerOptions
        {
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        })!.AsObject();

    private static string Text(JsonNode node, string property)
        => node[property]!.GetValue<string>();
    private static int Number(JsonNode node, string property)
        => node[property]!.GetValue<int>();
    private static double Double(JsonNode node, string property)
        => node[property]!.GetValue<double>();
}
