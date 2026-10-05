using System.Text.Json;
using System.Text.Json.Nodes;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.Gemini.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class GeminiInteractionsTests
{
    [Fact]
    public void RequestUsesTypedStepsFlatFunctionToolsAndStatelessDefault()
    {
        var request = new UnifiedRequest
        {
            Model = "gemini-3.6-flash",
            Instructions = "Be concise.",
            Messages = { new UnifiedMessage(MessageRole.User, "Weather?") },
            Tools = new List<ToolDefinition>
            {
                new()
                {
                    Name = "get_weather",
                    Description = "Get weather.",
                    Parameters = new Dictionary<string, object>
                    {
                        ["type"] = "object",
                        ["properties"] = new Dictionary<string, object>()
                    }
                }
            },
            ToolChoice = new ToolChoice
            {
                Type = ToolChoiceType.Specific,
                ToolName = "get_weather"
            }
        };

        var json = JsonSerializer.SerializeToNode(
            new GeminiInteractionsConverter().ConvertRequest(request))!.AsObject();

        Assert.Equal("gemini-3.6-flash", json["model"]!.GetValue<string>());
        Assert.False(json["store"]!.GetValue<bool>());
        Assert.Equal("Be concise.", json["system_instruction"]!.GetValue<string>());
        Assert.Equal("user_input", json["input"]![0]!["type"]!.GetValue<string>());
        Assert.Equal("function", json["tools"]![0]!["type"]!.GetValue<string>());
        Assert.Equal("get_weather", json["tools"]![0]!["name"]!.GetValue<string>());
        Assert.Equal("any", json["generation_config"]!["tool_choice"]!["allowed_tools"]!["mode"]!.GetValue<string>());
        Assert.Null(json["tool_choice"]);
    }

    [Fact]
    public void ResponseProjectsTextFunctionCallsUsageAndLosslessContinuation()
    {
        const string json = """
        {
          "id":"int_123",
          "model":"gemini-3.6-flash",
          "status":"requires_action",
          "steps":[
            {"type":"thought","signature":"sig","summary":[{"type":"text","text":"Check weather"}]},
            {"type":"model_output","content":[{"type":"text","text":"Checking."}]},
            {"type":"function_call","id":"call_1","name":"get_weather","arguments":{"city":"Boston"}}
          ],
          "usage":{"total_input_tokens":4,"total_output_tokens":5,"total_tokens":10,"total_thought_tokens":1,"total_tool_use_tokens":2}
        }
        """;
        var native = JsonSerializer.Deserialize<GeminiInteractionResponse>(json)!;

        var response = new GeminiInteractionsConverter().ConvertResponse(native);

        Assert.Equal("int_123", response.Id);
        Assert.Equal(4, response.Usage.InputTokens);
        Assert.Equal(1, response.Usage.ReasoningTokens);
        Assert.Equal(FinishReason.ToolCalls, Assert.Single(response.Choices).FinishReason);
        Assert.Collection(
            response.Choices[0].Message.Content,
            block => Assert.Equal("thought", Assert.IsType<ProviderNativeContent>(block).NativeType),
            block => Assert.Equal("Checking.", Assert.IsType<TextContent>(block).Text),
            block => Assert.Equal("call_1", Assert.IsType<ToolCallContent>(block).Id));
        Assert.Equal(3, response.Continuation!.NativeItems.Count);
    }

    [Fact]
    public void ServerManagedContinuationUsesPreviousInteractionIdAndNoReplay()
    {
        var request = new UnifiedRequest
        {
            Model = "gemini-3.6-flash",
            Messages = { new UnifiedMessage(MessageRole.User, "Next") },
            Continuation = new ProviderContinuationState
            {
                Provider = ProviderIds.Gemini,
                Mode = ContinuationMode.ServerManaged,
                ResponseId = "int_previous",
                NativeItems =
                {
                    JsonSerializer.SerializeToElement(new { type = "model_output" })
                }
            }
        };

        var converted = new GeminiInteractionsConverter().ConvertRequest(request);

        Assert.True(converted.Store);
        Assert.Equal("int_previous", converted.PreviousInteractionId);
        Assert.Single(converted.Input);
        Assert.Equal("user_input", converted.Input[0].GetProperty("type").GetString());
    }
}
