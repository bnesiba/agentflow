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
    public async Task Reasoning_controls_and_summary(Surface surface)
    {
        var request = Request(surface, "Find the least positive integer divisible by 12, 15 and 18. Reply with the number.");
        request.Parameters.MaxOutputTokens = 768;
        request.Reasoning = new()
        {
            Effort = ReasoningEffort.Medium,
            Output = ReasoningOutput.Summary,
            Anthropic = surface == Surface.Claude ? new() { Mode = AnthropicThinkingMode.Adaptive } : null,
            OpenAI = surface == Surface.OpenAI ? new() { Mode = OpenAIReasoningMode.Standard,
                Summary = OpenAIReasoningSummary.Auto } : null
        };
        var response = await Service(surface).GenerateAsync(request);
        var message = Assert.Single(response.Choices).Message;
        Assert.Contains("180", Text(message));
        // Providers may omit visible summaries. If one is returned it must have content.
        Assert.All(message.Content.OfType<ReasoningSummaryContent>(), summary =>
            Assert.False(string.IsNullOrWhiteSpace(summary.Text)));
        Assert.True(response.Usage.OutputTokens > 0);
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    public async Task Pro_mode_and_persisted_reasoning_context(Surface surface)
    {
        var request = Request(surface, "Calculate 23 times 17. Answer with the number.");
        request.Model = "gpt-5.6-luna";
        request.Parameters.MaxOutputTokens = 512;
        request.Reasoning = new()
        {
            Effort = ReasoningEffort.Low,
            OpenAI = new() { Mode = OpenAIReasoningMode.Pro, Context = OpenAIReasoningContext.AllTurns }
        };
        var response = await Service(surface).GenerateAsync(request);
        Assert.Contains("391", Text(Assert.Single(response.Choices).Message));
        Assert.NotEmpty(response.Continuation!.NativeItems);
    }

    [LiveTheory]
    [InlineData(Surface.Claude)]
    public async Task Claude_manual_thinking_budget(Surface surface)
    {
        var request = Request(surface, "Compute 17 times 19. Reply with the number.");
        request.Model = "claude-sonnet-4-6";
        request.Reasoning = new() { Anthropic = new() { Mode = AnthropicThinkingMode.Manual, BudgetTokens = 1024 } };
        request.Parameters.MaxOutputTokens = 1280;
        var response = await Service(surface).GenerateAsync(request);
        Assert.Contains("323", Text(Assert.Single(response.Choices).Message));
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Cancelling_an_open_stream_preserves_caller_cancellation(Surface surface)
    {
        using var cancellation = new CancellationTokenSource();
        var request = Request(surface, "List the integers from one to twenty.");
        await using var stream = Service(surface).StreamAsync(request, cancellation.Token).GetAsyncEnumerator();
        Assert.True(await stream.MoveNextAsync());
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await stream.MoveNextAsync());
    }
}
