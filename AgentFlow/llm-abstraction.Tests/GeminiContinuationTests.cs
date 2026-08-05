using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.Gemini.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class GeminiContinuationTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    [Fact]
    public void SignedPartsRoundTripWithoutMergingReorderingOrFieldLoss()
    {
        const string responseJson = """
        {
          "responseId": "gemini-response-123",
          "modelVersion": "gemini-3.5-flash",
          "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30,
            "thoughtsTokenCount": 5
          },
          "candidates": [
            {
              "index": 0,
              "finishReason": "STOP",
              "content": {
                "role": "model",
                "parts": [
                  {
                    "text": "internal summary",
                    "thought": true,
                    "thoughtSignature": "thought-part-signature",
                    "futureThoughtField": { "level": 2 }
                  },
                  {
                    "text": "I will check the weather.",
                    "thoughtSignature": "text-part-signature",
                    "futureTextField": [1, 2, 3]
                  },
                  {
                    "functionCall": {
                      "id": "call-123",
                      "name": "get_weather",
                      "args": { "city": "Boston" }
                    },
                    "thoughtSignature": "function-part-signature",
                    "futureFunctionField": "preserve-me"
                  },
                  {
                    "executableCode": {
                      "language": "PYTHON",
                      "code": "print('future part')"
                    },
                    "thoughtSignature": "unknown-part-signature"
                  }
                ]
              }
            }
          ]
        }
        """;

        var nativeResponse = JsonSerializer.Deserialize<GeminiGenerateResponse>(responseJson, JsonOptions)!;
        var converter = new GeminiConverter();
        var unified = converter.ConvertResponse(nativeResponse);
        var assistant = Assert.Single(unified.Choices).Message;

        Assert.Collection(
            assistant.Content,
            block =>
            {
                var thought = Assert.IsType<ProviderNativeContent>(block);
                Assert.Equal("thought", thought.NativeType);
                Assert.Equal("thought-part-signature", GetSignature(thought));
            },
            block =>
            {
                var text = Assert.IsType<TextContent>(block);
                Assert.Equal("I will check the weather.", text.Text);
                Assert.Equal("text-part-signature", GetSignature(text));
            },
            block =>
            {
                var call = Assert.IsType<ToolCallContent>(block);
                Assert.Equal("call-123", call.Id);
                Assert.Equal("get_weather", call.Name);
                Assert.Equal("function-part-signature", GetSignature(call));
            },
            block =>
            {
                var unknown = Assert.IsType<ProviderNativeContent>(block);
                Assert.Equal("executableCode", unknown.NativeType);
                Assert.Equal("unknown-part-signature", GetSignature(unknown));
            });

        var followUp = new UnifiedRequest
        {
            Model = "gemini-3.5-flash",
            Messages =
            {
                assistant,
                new UnifiedMessage(MessageRole.Tool, new List<ContentBlock>
                {
                    new ToolResultContent
                    {
                        ToolCallId = "call-123",
                        ToolName = "get_weather",
                        Output = new Dictionary<string, object> { ["temperature"] = 72 }
                    }
                })
            }
        };

        var replay = converter.ConvertRequest(followUp);
        var replayJson = JsonSerializer.SerializeToNode(replay, JsonOptions)!.AsObject();
        var replayParts = replayJson["contents"]![0]!["parts"]!.AsArray();
        var originalParts = JsonNode.Parse(responseJson)!["candidates"]![0]!["content"]!["parts"]!.AsArray();

        Assert.Equal(originalParts.Count, replayParts.Count);
        for (var index = 0; index < originalParts.Count; index++)
        {
            Assert.True(
                JsonNode.DeepEquals(originalParts[index], replayParts[index]),
                $"Gemini part {index} changed during replay.\nExpected: {originalParts[index]}\nActual: {replayParts[index]}");
        }
    }

    [Fact]
    public void MultipleSignedTextPartsRemainSeparateAndOrdered()
    {
        const string responseJson = """
        {
          "candidates": [{
            "content": {
              "role": "model",
              "parts": [
                { "text": "first", "thoughtSignature": "sig-first" },
                { "text": "second", "thoughtSignature": "sig-second" }
              ]
            }
          }]
        }
        """;

        var response = JsonSerializer.Deserialize<GeminiGenerateResponse>(responseJson, JsonOptions)!;
        var assistant = Assert.Single(new GeminiConverter().ConvertResponse(response).Choices).Message;

        Assert.Collection(
            assistant.Content,
            block =>
            {
                Assert.Equal("first", Assert.IsType<TextContent>(block).Text);
                Assert.Equal("sig-first", GetSignature(block));
            },
            block =>
            {
                Assert.Equal("second", Assert.IsType<TextContent>(block).Text);
                Assert.Equal("sig-second", GetSignature(block));
            });
    }

    [Fact]
    public void NativeContentFromAnotherProviderIsNotSentToGemini()
    {
        using var nativeBlock = JsonDocument.Parse("""{"type":"thinking","signature":"claude-signature"}""");
        var request = new UnifiedRequest
        {
            Model = "gemini-3.5-flash",
            Messages =
            {
                new UnifiedMessage(MessageRole.User, new List<ContentBlock>
                {
                    new TextContent { Text = "Hello" },
                    new ProviderNativeContent
                    {
                        NativeType = "thinking",
                        NativeRepresentation = ProviderNativeRepresentation.Create(
                            ProviderIds.Claude,
                            nativeBlock.RootElement)
                    }
                })
            }
        };

        var exception = Assert.Throws<NotSupportedException>(
            () => new GeminiConverter().ConvertRequest(request));
        Assert.Contains("originated from Gemini", exception.Message);
    }

    private static string? GetSignature(ContentBlock block)
    {
        return block.NativeRepresentation?.Value
            .GetProperty("thoughtSignature")
            .GetString();
    }
}
