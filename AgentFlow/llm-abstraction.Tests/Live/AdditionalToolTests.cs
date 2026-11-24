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
    public async Task Required_function_and_failed_result_can_complete_a_turn(Surface surface)
    {
        var service = Service(surface);
        var request = Request(surface, "Look up code A. Reply with SUCCESS on success, or FAILED followed by the number on error.");
        request.Tools = [new FunctionTool { Name = "lookup", Description = "Look up a code.", Parameters = StringSchema("code") }];
        request.ToolChoice = new() { Type = ToolChoiceType.Required };
        var choice = Assert.Single((await service.GenerateAsync(request)).Choices);
        Assert.Equal(FinishReason.ToolCalls, choice.FinishReason);
        var call = Assert.Single(choice.Message.Content.OfType<ToolCallContent>());
        Assert.Equal("lookup", call.Name);
        request.Messages.Add(choice.Message);
        request.Messages.Add(new(MessageRole.Tool, [new ToolResultContent
        {
            ToolCallId = call.Id, ToolName = call.Name, IsError = true, Output = "473"
        }]));
        request.ToolChoice = new() { Type = ToolChoiceType.Auto };
        var answer = Text(Assert.Single((await service.GenerateAsync(request)).Choices).Message);
        Assert.Contains("FAILED", answer, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("473", answer);
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Image_search_returns_provider_attribution(Surface surface)
    {
        var request = Request(surface, "Use image search for a photo of the Eiffel Tower. Return one image result.");
        request.Reasoning!.Effort = ReasoningEffort.Low;
        request.Parameters.MaxOutputTokens = 512;
        request.Tools = [ProviderTools.WebSearch(new()
        {
            ContentTypes = WebSearchContentTypes.Image,
            OpenAI = surface == Surface.OpenAI ? new() { MaximumImageResults = 1, IncludeImageCaptions = true } : null
        })];
        var choice = Assert.Single((await Service(surface).GenerateAsync(request)).Choices);
        var results = choice.Message.Content.OfType<ProviderToolResultContent>().ToList();
        Assert.NotEmpty(results);
        if (surface == Surface.GeminiInteractions)
            Assert.Contains(results, result => !string.IsNullOrEmpty(result.SearchSuggestionsHtml));
        else
            Assert.Contains(results.SelectMany(b => b.Sources), source => !string.IsNullOrEmpty(source.ImageUrl));
    }
}
