using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class GeminiToolResultTests
{
    [Fact]
    public void ToolResultWithoutNameResolvesNameFromPriorCall()
    {
        var request = new UnifiedRequest
        {
            Model = "gemini-3.5-flash",
            Messages =
            {
                new UnifiedMessage(MessageRole.User, "Weather?"),
                new UnifiedMessage(MessageRole.Assistant, new List<ContentBlock>
                {
                    new ToolCallContent
                    {
                        Id = "call-weather",
                        Name = "get_weather",
                        Input = new Dictionary<string, object> { ["city"] = "Boston" }
                    }
                }),
                new UnifiedMessage(MessageRole.Tool, new List<ContentBlock>
                {
                    new ToolResultContent
                    {
                        ToolCallId = "call-weather",
                        Output = "72 F"
                    }
                })
            }
        };

        var converted = new GeminiConverter().ConvertRequest(request);
        var response = Assert.Single(converted.Contents[2].Parts).FunctionResponse!;

        Assert.Equal("get_weather", response.Name);
        Assert.Equal("call-weather", response.Id);
    }

    [Fact]
    public void OrphanedToolResultWithoutNameIsRejected()
    {
        var request = RequestWithToolResult(toolName: null);

        var error = Assert.Throws<ArgumentException>(
            () => new GeminiConverter().ConvertRequest(request));
        Assert.Contains("requires ToolName", error.Message);
    }

    [Fact]
    public void ExplicitNameThatConflictsWithPriorCallIsRejected()
    {
        var request = new UnifiedRequest
        {
            Model = "gemini-3.5-flash",
            Messages =
            {
                new UnifiedMessage(MessageRole.Assistant, new List<ContentBlock>
                {
                    new ToolCallContent { Id = "call-1", Name = "expected_tool" }
                }),
                new UnifiedMessage(MessageRole.Tool, new List<ContentBlock>
                {
                    new ToolResultContent
                    {
                        ToolCallId = "call-1",
                        ToolName = "different_tool",
                        Output = "done"
                    }
                })
            }
        };

        var error = Assert.Throws<ArgumentException>(
            () => new GeminiConverter().ConvertRequest(request));
        Assert.Contains("does not match", error.Message);
    }

    [Fact]
    public void OrphanedToolResultWithExplicitNameIsAccepted()
    {
        var converted = new GeminiConverter().ConvertRequest(
            RequestWithToolResult("get_weather"));

        Assert.Equal(
            "get_weather",
            Assert.Single(converted.Contents).Parts[0].FunctionResponse!.Name);
    }

    private static UnifiedRequest RequestWithToolResult(string? toolName)
    {
        return new UnifiedRequest
        {
            Model = "gemini-3.5-flash",
            Messages =
            {
                new UnifiedMessage(MessageRole.Tool, new List<ContentBlock>
                {
                    new ToolResultContent
                    {
                        ToolCallId = "orphaned-call",
                        ToolName = toolName,
                        Output = "done"
                    }
                })
            }
        };
    }
}
