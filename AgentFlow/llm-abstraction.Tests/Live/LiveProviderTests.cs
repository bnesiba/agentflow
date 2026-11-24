using LLMAbstraction.Core;
using LLMAbstraction.Core.Interfaces;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Transport;
using LLMAbstraction.Providers.Gemini;
using Xunit;

namespace LLMAbstraction.Tests.Live;

public enum Surface { OpenAI, Claude, GeminiInteractions, GeminiGenerateContent }

public sealed class LiveTheoryAttribute : TheoryAttribute
{
    public LiveTheoryAttribute()
    {
        if (Environment.GetEnvironmentVariable("AGENTFLOW_LIVE") != "1")
            Skip = "Set AGENTFLOW_LIVE=1 to enable billed provider calls.";
    }
}

[Trait("Category", "Live")]
public partial class LiveProviderTests
{
    private static string Key(Surface surface)
    {
        var name = surface switch
        {
            Surface.OpenAI => "OPENAI_API_KEY",
            Surface.Claude => "ANTHROPIC_API_KEY",
            Surface.GeminiInteractions => "GEMINI_API_KEY",
            Surface.GeminiGenerateContent => "GEMINI_API_KEY",
            _ => throw new ArgumentOutOfRangeException(nameof(surface), surface, "Unknown provider surface.")
        };
        return Environment.GetEnvironmentVariable(name) is { Length: > 0 } key
            ? key : throw new InvalidOperationException($"Set {name} to run the selected live tests.");
    }

    private static ILLMServiceWithTokenCounting Service(Surface surface, LLMTransportOptions? transport = null) => surface switch
    {
        Surface.OpenAI => LLMServiceFactory.CreateOpenAI(Key(surface), transport: transport),
        Surface.Claude => LLMServiceFactory.CreateClaude(Key(surface), transport: transport),
        Surface.GeminiInteractions => LLMServiceFactory.CreateGemini(Key(surface), transport: transport),
        Surface.GeminiGenerateContent => new GeminiService(Key(surface), apiMode: GeminiApiMode.GenerateContent, transportOptions: transport),
        _ => throw new ArgumentOutOfRangeException(nameof(surface))
    };

    private static UnifiedRequest Request(Surface surface, string prompt) => new()
    {
        Model = surface switch
        {
            Surface.OpenAI => "gpt-6-luna",
            Surface.Claude => "claude-haiku-5-5",
            Surface.GeminiInteractions => "gemini-3.5-flash-lite",
            Surface.GeminiGenerateContent => "gemini-3.5-flash-lite",
            _ => throw new ArgumentOutOfRangeException(nameof(surface), surface, "Unknown provider surface.")
        },
        Messages = [new(MessageRole.User, prompt)],
        Parameters = new() { MaxOutputTokens = 128 },
        Reasoning = new() { Effort = surface == Surface.OpenAI ? ReasoningEffort.None : ReasoningEffort.Low }
    };

    private static string Text(UnifiedMessage message) =>
        string.Concat(message.Content.OfType<TextContent>().Select(block => block.Text)).Trim();

    private static Dictionary<string, object> StringSchema(string field) => new()
    {
        ["type"] = "object",
        ["properties"] = new Dictionary<string, object> { [field] = new { type = "string" } },
        ["required"] = new[] { field },
        ["additionalProperties"] = false
    };
}
