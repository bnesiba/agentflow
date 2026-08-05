using System.Text.Json;
using System.Text.Json.Nodes;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Claude.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class ClaudeContinuationTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
    };

    [Fact]
    public void ThinkingToolRoundTripPreservesEveryNativeBlockInOrder()
    {
        const string responseJson = """
        {
          "id": "msg_123",
          "type": "message",
          "role": "assistant",
          "model": "claude-sonnet-4-6",
          "stop_reason": "tool_use",
          "usage": { "input_tokens": 12, "output_tokens": 34 },
          "content": [
            {
              "type": "thinking",
              "thinking": "summary one",
              "signature": "sig-one",
              "future_thinking_field": { "nested": true }
            },
            {
              "type": "thinking",
              "thinking": "summary two",
              "signature": "sig-two"
            },
            {
              "type": "redacted_thinking",
              "data": "encrypted-redacted-data",
              "future_redaction_field": 7
            },
            {
              "type": "text",
              "text": "I need to inspect the weather.",
              "citations": [{ "type": "future_citation", "value": "citation-data" }]
            },
            {
              "type": "tool_use",
              "id": "toolu_123",
              "name": "get_weather",
              "input": { "city": "Boston" },
              "caller": { "type": "future-caller" }
            }
          ]
        }
        """;

        var nativeResponse = JsonSerializer.Deserialize<ClaudeMessageResponse>(responseJson, JsonOptions)!;
        var converter = new ClaudeConverter();

        var unified = converter.ConvertResponse(nativeResponse);
        var assistant = Assert.Single(unified.Choices).Message;

        Assert.Collection(
            assistant.Content,
            block => AssertNativeBlock(block, "thinking", "sig-one"),
            block => AssertNativeBlock(block, "thinking", "sig-two"),
            block => AssertNativeBlock(block, "redacted_thinking", expectedSignature: null),
            block => Assert.IsType<TextContent>(block),
            block => Assert.IsType<ToolCallContent>(block));

        var followUp = new UnifiedRequest
        {
            Model = "claude-sonnet-4-6",
            Messages =
            {
                assistant,
                new UnifiedMessage(MessageRole.Tool, new List<ContentBlock>
                {
                    new ToolResultContent
                    {
                        ToolCallId = "toolu_123",
                        Output = new Dictionary<string, object> { ["temperature"] = 72 }
                    }
                })
            }
        };

        var replay = converter.ConvertRequest(followUp);
        var replayJson = JsonSerializer.SerializeToNode(replay, JsonOptions)!.AsObject();
        var replayBlocks = replayJson["messages"]![0]!["content"]!.AsArray();
        var originalBlocks = JsonNode.Parse(responseJson)!["content"]!.AsArray();

        Assert.Equal(originalBlocks.Count, replayBlocks.Count);
        for (var index = 0; index < originalBlocks.Count; index++)
        {
            Assert.True(
                JsonNode.DeepEquals(originalBlocks[index], replayBlocks[index]),
                $"Claude content block {index} changed during response-to-request replay.\nExpected: {originalBlocks[index]}\nActual: {replayBlocks[index]}");
        }
    }

    [Fact]
    public void NativeContentFromAnotherProviderIsNotSentToClaude()
    {
        using var document = JsonDocument.Parse("""{"type":"thought","thoughtSignature":"gemini-signature"}""");
        var request = new UnifiedRequest
        {
            Model = "claude-sonnet-4-6",
            Messages =
            {
                new UnifiedMessage(MessageRole.User, new List<ContentBlock>
                {
                    new TextContent { Text = "Hello" },
                    new ProviderNativeContent
                    {
                        NativeType = "thought",
                        NativeRepresentation = ProviderNativeRepresentation.Create(
                            ProviderIds.Gemini,
                            document.RootElement)
                    }
                })
            }
        };

        var exception = Assert.Throws<NotSupportedException>(
            () => new ClaudeConverter().ConvertRequest(request));
        Assert.Contains("originated from Claude", exception.Message);
    }

    private static void AssertNativeBlock(
        ContentBlock block,
        string expectedType,
        string? expectedSignature)
    {
        var native = Assert.IsType<ProviderNativeContent>(block);
        Assert.Equal(expectedType, native.NativeType);
        Assert.Equal(ProviderIds.Claude, native.Provider);

        if (expectedSignature != null)
        {
            Assert.Equal(
                expectedSignature,
                native.NativeRepresentation!.Value.GetProperty("signature").GetString());
        }
    }
}
