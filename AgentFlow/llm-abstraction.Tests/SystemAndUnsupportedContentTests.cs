using System.Text.Json;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class SystemAndUnsupportedContentTests
{
    [Fact]
    public void ClaudeCombinesInstructionsAndSystemMessagesAndRemovesSystemTurns()
    {
        var request = CreateRequest("claude-sonnet-4-6");
        var converted = new ClaudeConverter().ConvertRequest(request);

        Assert.Equal("Base instructions\n\nFirst system block\nSecond system block", converted.System);
        Assert.Single(converted.Messages);
        Assert.Equal("user", converted.Messages[0].Role);
    }

    [Fact]
    public void OpenAICombinesInstructionsAndSystemMessagesAndRemovesSystemInputItems()
    {
        var request = CreateRequest("gpt-5.6");
        var converted = new OpenAIConverter().ConvertRequest(request);

        Assert.Equal("Base instructions\n\nFirst system block\nSecond system block", converted.Instructions);
        var json = JsonSerializer.Serialize(converted.Input);
        Assert.DoesNotContain("First system block", json);
        Assert.Contains("User content", json);
    }

    [Fact]
    public void GeminiCombinesInstructionsAndSystemMessagesAndRemovesSystemTurns()
    {
        var request = CreateRequest("gemini-3.5-flash");
        var converted = new GeminiConverter().ConvertRequest(request);

        Assert.Equal(
            "Base instructions\n\nFirst system block\nSecond system block",
            Assert.Single(converted.SystemInstruction!.Parts).Text);
        Assert.Single(converted.Contents);
        Assert.Equal("user", converted.Contents[0].Role);
    }

    [Fact]
    public void NonTextSystemContentIsRejectedForEveryProvider()
    {
        var request = new UnifiedRequest
        {
            Model = "unused",
            Messages =
            {
                new UnifiedMessage(MessageRole.System, new List<ContentBlock>
                {
                    new ImageContent { Source = new ImageSource { Url = "https://example.com/image.png" } }
                }),
                new UnifiedMessage(MessageRole.User, "Hello")
            }
        };

        Assert.Throws<NotSupportedException>(() => new ClaudeConverter().ConvertRequest(request));
        Assert.Throws<NotSupportedException>(() => new OpenAIConverter().ConvertRequest(request));
        Assert.Throws<NotSupportedException>(() => new GeminiConverter().ConvertRequest(request));
    }

    [Fact]
    public void UnsupportedOrMissingMediaSourcesAreRejectedInsteadOfDropped()
    {
        var missingClaudeImage = RequestWithContent("claude-sonnet-4-6", new ImageContent());
        Assert.Throws<NotSupportedException>(() => new ClaudeConverter().ConvertRequest(missingClaudeImage));

        var geminiUrlImage = RequestWithContent("gemini-3.5-flash", new ImageContent
        {
            Source = new ImageSource { Url = "https://example.com/image.png" }
        });
        Assert.Throws<NotSupportedException>(() => new GeminiConverter().ConvertRequest(geminiUrlImage));

        var openAIFileUri = RequestWithContent("gpt-5.6", new MediaContent
        {
            MediaType = "image/png",
            Source = new MediaSource { FileUri = "files/provider-only" }
        });
        Assert.Throws<NotSupportedException>(() => new OpenAIConverter().ConvertRequest(openAIFileUri));
    }

    private static UnifiedRequest CreateRequest(string model)
    {
        return new UnifiedRequest
        {
            Model = model,
            Instructions = "Base instructions",
            Messages =
            {
                new UnifiedMessage(MessageRole.System, new List<ContentBlock>
                {
                    new TextContent { Text = "First system block" },
                    new TextContent { Text = "Second system block" }
                }),
                new UnifiedMessage(MessageRole.User, "User content")
            }
        };
    }

    private static UnifiedRequest RequestWithContent(string model, ContentBlock content)
    {
        return new UnifiedRequest
        {
            Model = model,
            Messages = { new UnifiedMessage(MessageRole.User, new List<ContentBlock> { content }) }
        };
    }
}
