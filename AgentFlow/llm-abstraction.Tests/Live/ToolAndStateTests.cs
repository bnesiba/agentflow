using LLMAbstraction.Core.Models;
using Xunit;
namespace LLMAbstraction.Tests.Live;

public partial class LiveProviderTests
{
    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Streamed_function_call_replays_with_tool_result(Surface surface)
    {
        var service = Service(surface);
        var request = Request(surface, "Call lookup for key amber. Reply with the returned code.");
        request.Tools = [new FunctionTool
        {
            Name = "lookup", Description = "Look up a code by key.", Parameters = StringSchema("key"),
            Strict = surface is Surface.Claude or Surface.OpenAI ? true : null
        }];
        request.ToolChoice = new() { Type = ToolChoiceType.Specific, ToolName = "lookup",
            DisableParallelToolUse = surface is Surface.Claude or Surface.OpenAI ? true : null };
        var chunks = new List<StreamChunk>();
        await foreach (var chunk in service.StreamAsync(request))
            chunks.Add(chunk);
        Assert.Contains(chunks, c => c.Delta.ToolCalls?.Count > 0);
        var assistant = Assert.Single(chunks, c => c.CompletedMessage != null).CompletedMessage!;
        var call = Assert.Single(assistant.Content.OfType<ToolCallContent>());
        Assert.Equal("lookup", call.Name);
        Assert.Equal("amber", call.Input["key"].ToString());
        Assert.False(string.IsNullOrWhiteSpace(call.Id));

        request.Messages.Add(assistant);
        request.Messages.Add(new(MessageRole.Tool, [new ToolResultContent
        {
            ToolCallId = call.Id, ToolName = call.Name, Output = "CODE-417"
        }]));
        request.ToolChoice = new() { Type = ToolChoiceType.None };
        var reply = Assert.Single((await service.GenerateAsync(request)).Choices);
        Assert.Contains("417", Text(reply.Message));
        Assert.Empty(reply.Message.Content.OfType<ToolCallContent>());
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.GeminiInteractions)]
    public async Task Stored_turn_remembers_user_input(Surface surface)
    {
        var service = Service(surface);
        var request = Request(surface, "Remember the token ALDER-593. Reply OK.");
        request.ProviderOptions = surface == Surface.OpenAI
            ? new() { OpenAI = new() { ["store"] = true } }
            : new() { Gemini = new() { ["store"] = true } };
        var first = await service.GenerateAsync(request);
        Assert.False(string.IsNullOrWhiteSpace(first.Continuation?.ResponseId));
        var next = Request(surface, "What token did I give you?");
        next.Continuation = first.Continuation;
        next.Continuation!.Mode = ContinuationMode.ServerManaged;
        Assert.Contains("ALDER-593", Text(Assert.Single((await service.GenerateAsync(next)).Choices).Message));
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.GeminiInteractions)]
    public async Task Stateless_continuation_replays_native_items(Surface surface)
    {
        var service = Service(surface);
        var request = Request(surface, "Say only FERN-729.");
        request.ProviderOptions = surface == Surface.OpenAI
            ? new() { OpenAI = new() { ["store"] = false } }
            : new() { Gemini = new() { ["store"] = false } };
        var response = await service.GenerateAsync(request);
        Assert.Contains("FERN-729", Text(Assert.Single(response.Choices).Message));
        Assert.NotEmpty(response.Continuation!.NativeItems);
        var next = Request(surface, "Repeat exactly what you just said.");
        next.ProviderOptions = request.ProviderOptions;
        next.Continuation = response.Continuation;
        next.Continuation.Mode = ContinuationMode.StatelessReplay;
        Assert.Contains("FERN-729", Text(Assert.Single((await service.GenerateAsync(next)).Choices).Message));
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Hosted_search_returns_attributed_web_evidence(Surface surface)
    {
        var request = Request(surface, "Search the web for the latest NASA image of the day. Give its title and cite the source.");
        request.Parameters.MaxOutputTokens = surface == Surface.Claude ? 1536 : 512;
        request.Reasoning!.Effort = ReasoningEffort.Low;
        request.Tools = [ProviderTools.WebSearch(new()
        {
            AllowedDomains = surface is Surface.OpenAI or Surface.Claude ? ["nasa.gov"] : null,
            Location = surface is Surface.OpenAI or Surface.Claude ? new() { Country = "US" } : null,
            Gemini = surface == Surface.GeminiGenerateContent ? new()
            {
                StartTime = DateTimeOffset.UtcNow.AddDays(-30), EndTime = DateTimeOffset.UtcNow
            } : null,
            OpenAI = surface == Surface.OpenAI ? new() { ContextSize = WebSearchContextSize.Low, IncludeAllSources = true } : null,
            Anthropic = surface == Surface.Claude ? new() { MaximumUses = 1 } : null
        })];
        var choice = Assert.Single((await Service(surface).GenerateAsync(request)).Choices);
        Assert.Contains(choice.Message.Evidence.Sources, s => Uri.TryCreate(s.Uri, UriKind.Absolute, out _));
        Assert.Contains(choice.Message.Content, b => b is ProviderToolCallContent or ProviderToolResultContent);
        Assert.True(!string.IsNullOrWhiteSpace(Text(choice.Message)), $"Search returned no answer text (finish reason: {choice.FinishReason}).");
    }
}
