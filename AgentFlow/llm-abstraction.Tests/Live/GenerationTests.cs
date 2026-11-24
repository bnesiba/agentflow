using LLMAbstraction.Core.Models;
using Xunit;
using System.Text.Json;
using LLMAbstraction.Core.Errors;
using LLMAbstraction.Core.Transport;

namespace LLMAbstraction.Tests.Live;

public partial class LiveProviderTests
{
    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Instructions_counting_usage_and_transport(Surface surface)
    {
        var admission = new RecordingAdmission();
        var service = Service(surface, new() { AdmissionPolicy = admission });
        var request = Request(surface, "What is the passphrase?");
        request.Messages.Insert(0, new(MessageRole.System, "The passphrase is cobalt."));
        request.Instructions = "Reply only with the passphrase.";
        request.Metadata = surface switch
        {
            Surface.OpenAI => new() { UserId = "live-test", Tags = new() { ["suite"] = "integration" } },
            Surface.Claude => new() { UserId = "live-test" },
            _ => null
        };

        var count = await service.CountInputTokensAsync(request);
        Assert.True(count.InputTokens > 0);
        Assert.Equal(surface is Surface.Claude or Surface.GeminiInteractions
            ? TokenCountAccuracy.Estimate : TokenCountAccuracy.Exact, count.Accuracy);
        request.Transport = new() { EstimatedInputTokens = count.InputTokens };
        request.ProviderOptions = surface switch
        {
            Surface.OpenAI => new() { OpenAI = new() { ["service_tier"] = "default" } },
            Surface.Claude => new() { Claude = new() { ["service_tier"] = "standard_only" } },
            Surface.GeminiInteractions => new() { Gemini = new() { ["service_tier"] = "standard" } },
            _ => new() { Gemini = new() { ["safetySettings"] = new[]
            {
                new { category = "HARM_CATEGORY_HARASSMENT", threshold = "BLOCK_MEDIUM_AND_ABOVE" }
            } } }
        };
        var response = await service.GenerateAsync(request);
        Assert.Equal(2, admission.Calls.Count);
        Assert.Equal(count.InputTokens, admission.Calls[1].EstimatedInputTokens);
        Assert.Equal(surface switch
        {
            Surface.OpenAI => "responses",
            Surface.Claude => "/v1/messages",
            Surface.GeminiInteractions => "interactions",
            _ => $"models/{request.Model}:generateContent"
        }, admission.Calls[1].Endpoint);
        var choice = Assert.Single(response.Choices);
        Assert.Equal(FinishReason.Stop, choice.FinishReason);
        Assert.Contains("cobalt", Text(choice.Message), StringComparison.OrdinalIgnoreCase);
        Assert.True(response.Usage.InputTokens > 0 && response.Usage.OutputTokens > 0);
        Assert.InRange(response.Usage.OutputTokens, 1, request.Parameters.MaxOutputTokens!.Value);
        Assert.Equal(System.Net.HttpStatusCode.OK, Assert.Single(response.Transport!.Attempts).StatusCode);
    }

    private sealed class RecordingAdmission : IRequestAdmissionPolicy
    {
        public List<RequestAdmissionContext> Calls { get; } = [];
        public ValueTask<IDisposable?> AcquireAsync(RequestAdmissionContext context, CancellationToken cancellationToken)
        {
            Calls.Add(context);
            return ValueTask.FromResult<IDisposable?>(null);
        }
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Output_cap_is_reported_as_truncation(Surface surface)
    {
        var request = Request(surface, "Write every integer from 1 to 200 in order, separated by spaces. Do not abbreviate.");
        request.Parameters.MaxOutputTokens = 16;
        var response = await Service(surface).GenerateAsync(request);
        Assert.Equal(FinishReason.MaxTokens, Assert.Single(response.Choices).FinishReason);
        Assert.InRange(response.Usage.OutputTokens, 0, 16);
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Stream_delivers_text_and_completed_message(Surface surface)
    {
        var chunks = new List<StreamChunk>();
        await foreach (var chunk in Service(surface).StreamAsync(Request(surface, "Reply only with cobalt.")))
            chunks.Add(chunk);
        var terminal = Assert.Single(chunks, c => c.CompletedMessage != null);
        Assert.Equal(FinishReason.Stop, terminal.FinishReason);
        var text = string.Concat(chunks.Select(c => c.Delta.Content)).Trim();
        Assert.Contains("cobalt", text, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(text, Text(terminal.CompletedMessage!));
        Assert.Contains(chunks, c => c.Usage?.OutputTokens > 0);
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Json_schema(Surface surface)
    {
        var request = Request(surface, "Use teal as the value.");
        request.Output = new() { Kind = OutputFormatKind.JsonSchema,
            JsonSchema = new() { Name = "color", Schema = StringSchema("color") } };
        using var json = JsonDocument.Parse(Text(Assert.Single((await Service(surface).GenerateAsync(request)).Choices).Message));
        Assert.Equal("teal", json.RootElement.GetProperty("color").GetString());
        Assert.Single(json.RootElement.EnumerateObject());
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Json_object(Surface surface)
    {
        var request = Request(surface, "Return a JSON object with the integer n equal to 7.");
        request.Output = new() { Kind = OutputFormatKind.JsonObject };
        using var json = JsonDocument.Parse(Text(Assert.Single((await Service(surface).GenerateAsync(request)).Choices).Message));
        Assert.Equal(7, json.RootElement.GetProperty("n").GetInt32());
    }

    [LiveTheory]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Stop_sequence_truncates_text(Surface surface)
    {
        var request = Request(surface, "Copy exactly: red END blue");
        request.Parameters.StopSequences = ["END"];
        Assert.Equal("red", Text(Assert.Single((await Service(surface).GenerateAsync(request)).Choices).Message),
            ignoreCase: true);
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Invalid_model_returns_normalized_provider_error(Surface surface)
    {
        var request = Request(surface, "Hi");
        request.Model = "agentflow-nonexistent-model";
        request.Reasoning = null;
        var error = await Assert.ThrowsAsync<LLMApiException>(() => Service(surface).GenerateAsync(request));
        Assert.True(error.StatusCode is System.Net.HttpStatusCode.BadRequest or System.Net.HttpStatusCode.NotFound);
        Assert.False(string.IsNullOrWhiteSpace(error.RawResponse));
    }
}
