using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.OpenAI.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class OpenAIContinuationTests
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    [Fact]
    public void StatelessContinuationPreservesAndReplaysAllOutputItemsInOrder()
    {
        const string responseJson = """
        {
          "id": "resp_123",
          "model": "gpt-5.6",
          "status": "completed",
          "usage": { "input_tokens": 10, "output_tokens": 20, "total_tokens": 30 },
          "output": [
            {
              "id": "rs_123",
              "type": "reasoning",
              "status": "completed",
              "summary": [{ "type": "summary_text", "text": "I should call the tool." }],
              "encrypted_content": "opaque-encrypted-reasoning",
              "future_reasoning_field": { "enabled": true }
            },
            {
              "id": "msg_123",
              "type": "message",
              "status": "completed",
              "role": "assistant",
              "content": [
                {
                  "type": "output_text",
                  "text": "Checking now.",
                  "annotations": [{ "type": "future_annotation", "value": 3 }]
                }
              ]
            },
            {
              "id": "fc_123",
              "type": "function_call",
              "status": "completed",
              "call_id": "call_123",
              "name": "get_weather",
              "arguments": "{\"city\":\"Boston\"}",
              "caller": "direct"
            }
          ]
        }
        """;

        var nativeResponse = JsonSerializer.Deserialize<OpenAIResponse>(responseJson, JsonOptions)!;
        var converter = new OpenAIConverter();
        var unified = converter.ConvertResponse(nativeResponse);

        Assert.Equal("resp_123", unified.Continuation!.ResponseId);
        Assert.Equal(ProviderIds.OpenAI, unified.Continuation.Provider);
        Assert.Equal(3, unified.Continuation.NativeItems.Count);

        var originalOutput = JsonNode.Parse(responseJson)!["output"]!.AsArray();
        for (var index = 0; index < originalOutput.Count; index++)
        {
            var retained = JsonNode.Parse(unified.Continuation.NativeItems[index].GetRawText());
            Assert.True(
                JsonNode.DeepEquals(originalOutput[index], retained),
                $"OpenAI output item {index} changed while being retained.\nExpected: {originalOutput[index]}\nActual: {retained}");
        }

        var toolCall = Assert.IsType<ToolCallContent>(
            Assert.Single(unified.Choices).Message.Content.Last());
        Assert.Equal("call_123", toolCall.Id);

        var nextRequest = new UnifiedRequest
        {
            Model = "gpt-5.6",
            Continuation = unified.Continuation,
            Messages =
            {
                new UnifiedMessage(MessageRole.Tool, new List<ContentBlock>
                {
                    new ToolResultContent
                    {
                        ToolCallId = "call_123",
                        Output = "72 F"
                    }
                })
            }
        };

        var replay = converter.ConvertRequest(nextRequest);
        Assert.Null(replay.PreviousResponseId);
        Assert.Equal(4, replay.Input.Count);

        var replayJson = JsonSerializer.SerializeToNode(replay, JsonOptions)!.AsObject();
        var replayInput = replayJson["input"]!.AsArray();
        for (var index = 0; index < originalOutput.Count; index++)
        {
            Assert.True(JsonNode.DeepEquals(originalOutput[index], replayInput[index]));
        }

        Assert.Equal("function_call_output", replayInput[3]!["type"]!.GetValue<string>());
        Assert.Equal("call_123", replayInput[3]!["call_id"]!.GetValue<string>());
    }

    [Fact]
    public void ServerManagedContinuationUsesPreviousResponseIdWithoutReplayingItems()
    {
        using var nativeItem = JsonDocument.Parse("""{"type":"reasoning","id":"rs_should_not_replay"}""");
        var request = new UnifiedRequest
        {
            Model = "gpt-5.6",
            Continuation = new ProviderContinuationState
            {
                Provider = ProviderIds.OpenAI,
                Mode = ContinuationMode.ServerManaged,
                ResponseId = "resp_previous",
                NativeItems = { nativeItem.RootElement.Clone() }
            },
            Messages = { new UnifiedMessage(MessageRole.User, "Continue") }
        };

        var converted = new OpenAIConverter().ConvertRequest(request);
        var json = JsonSerializer.Serialize(converted, JsonOptions);

        Assert.Equal("resp_previous", converted.PreviousResponseId);
        Assert.Single(converted.Input);
        Assert.DoesNotContain("rs_should_not_replay", json);
    }

    [Fact]
    public void ContinuationFromAnotherProviderIsNotSentToOpenAI()
    {
        using var nativeItem = JsonDocument.Parse("""{"type":"thinking","signature":"claude-secret"}""");
        var request = new UnifiedRequest
        {
            Model = "gpt-5.6",
            Continuation = new ProviderContinuationState
            {
                Provider = ProviderIds.Claude,
                NativeItems = { nativeItem.RootElement.Clone() }
            },
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
        };

        // Response-level continuation from another provider is deliberately
        // ignored; it is not a content block and should not prevent a portable
        // user message from being sent.
        var converted = new OpenAIConverter().ConvertRequest(request);
        Assert.Null(converted.PreviousResponseId);
        Assert.Single(converted.Input);
    }

    [Fact]
    public async Task CompletedStreamCarriesReplayableContinuationState()
    {
        const string eventJson = """
        {
          "type": "response.completed",
          "response": {
            "id": "resp_streamed",
            "model": "gpt-5.6",
            "status": "completed",
            "output": [
              {
                "id": "rs_streamed",
                "type": "reasoning",
                "status": "completed",
                "encrypted_content": "streamed-encrypted-content"
              }
            ],
            "usage": { "input_tokens": 1, "output_tokens": 2, "total_tokens": 3 }
          }
        }
        """;
        var compactEventJson = JsonNode.Parse(eventJson)!.ToJsonString();
        var sse = $"data: {compactEventJson}\n\n";
        using var httpClient = new HttpClient(new TestHttpMessageHandler(sse))
        {
            BaseAddress = new Uri("https://api.openai.com/v1/")
        };
        var service = new OpenAIService(httpClient, "test-key");

        var chunks = new List<StreamChunk>();
        await foreach (var chunk in service.StreamAsync(new UnifiedRequest
        {
            Model = "gpt-5.6",
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
        }))
        {
            chunks.Add(chunk);
        }

        var completed = Assert.Single(chunks);
        Assert.Equal("resp_streamed", completed.Continuation!.ResponseId);
        var item = Assert.Single(completed.Continuation.NativeItems);
        Assert.Equal(
            "streamed-encrypted-content",
            item.GetProperty("encrypted_content").GetString());
    }
}
