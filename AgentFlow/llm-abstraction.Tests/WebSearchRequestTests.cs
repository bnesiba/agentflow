using System.Text.Json;
using System.Text.Json.Nodes;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class WebSearchRequestTests
{
    [Fact]
    public void OpenAIWebSearchMapsPortableAndNativeOptions()
    {
        var request = Request("gpt-5.6", ProviderTools.WebSearch(new WebSearchOptions
        {
            AllowedDomains = new[] { "example.com" },
            BlockedDomains = new[] { "blocked.example.com" },
            Location = new ApproximateLocation { Country = "US", Timezone = "America/New_York" },
            ContentTypes = WebSearchContentTypes.Web | WebSearchContentTypes.Image,
            OpenAI = new OpenAIWebSearchOptions
            {
                ContextSize = WebSearchContextSize.High,
                IncludeAllSources = true,
                UnlimitedReturnTokenBudget = true,
                ExternalWebAccess = false,
                MaximumImageResults = 3,
                IncludeImageCaptions = true
            }
        }));

        var json = Serialize(new OpenAIConverter().ConvertRequest(request));
        var tool = json["tools"]![0]!;

        Assert.Equal("web_search", Text(tool, "type"));
        Assert.Equal("example.com", tool["filters"]!["allowed_domains"]![0]!.GetValue<string>());
        Assert.Equal("US", Text(tool["user_location"]!, "country"));
        Assert.Equal("high", Text(tool, "search_context_size"));
        Assert.Equal("unlimited", Text(tool, "return_token_budget"));
        Assert.False(tool["external_web_access"]!.GetValue<bool>());
        Assert.Equal("image", tool["search_content_types"]![1]!.GetValue<string>());
        Assert.Equal(3, tool["image_settings"]!["max_results"]!.GetValue<int>());
        Assert.Equal("web_search_call.action.sources", json["include"]![0]!.GetValue<string>());
        Assert.Equal("web_search_call.results", json["include"]![1]!.GetValue<string>());
    }

    [Fact]
    public void AnthropicWebSearchSelectsVersionAndMapsControls()
    {
        var request = Request("claude-opus-4-8", ProviderTools.WebSearch(new WebSearchOptions
        {
            AllowedDomains = new[] { "example.com" },
            Location = new ApproximateLocation { City = "Boston", Country = "US" },
            Anthropic = new AnthropicWebSearchOptions
            {
                MaximumUses = 4,
                DynamicFiltering = false,
                IncludeFullResults = false
            }
        }));

        var json = Serialize(new ClaudeConverter().ConvertRequest(request));
        var tool = json["tools"]![0]!;

        Assert.Equal("web_search_20260318", Text(tool, "type"));
        Assert.Equal("web_search", Text(tool, "name"));
        Assert.Equal(4, tool["max_uses"]!.GetValue<int>());
        Assert.Equal("example.com", tool["allowed_domains"]![0]!.GetValue<string>());
        Assert.Equal("Boston", Text(tool["user_location"]!, "city"));
        Assert.Equal("direct", tool["allowed_callers"]![0]!.GetValue<string>());
        Assert.Equal("excluded", Text(tool, "response_inclusion"));
    }

    [Fact]
    public void GeminiInteractionsMapsWebAndImageSearchTypes()
    {
        var request = Request("gemini-3.6-flash", ProviderTools.WebSearch(new WebSearchOptions
        {
            ContentTypes = WebSearchContentTypes.Web | WebSearchContentTypes.Image
        }));

        var json = Serialize(new GeminiInteractionsConverter().ConvertRequest(request));
        var tool = json["tools"]![0]!;

        Assert.Equal("google_search", Text(tool, "type"));
        Assert.Equal("web_search", tool["search_types"]![0]!.GetValue<string>());
        Assert.Equal("image_search", tool["search_types"]![1]!.GetValue<string>());
    }

    [Fact]
    public void SpecificWebSearchChoiceMapsToEachProvidersNativeIdentity()
    {
        var openAITool = ProviderTools.WebSearch();
        openAITool.Id = "search_current_web";
        var openAI = Request("gpt-5.6", openAITool);
        openAI.ToolChoice = new ToolChoice { Type = ToolChoiceType.Specific, ToolName = "search_current_web" };
        var openAIJson = Serialize(new OpenAIConverter().ConvertRequest(openAI));
        Assert.Equal("web_search", Text(openAIJson["tool_choice"]!, "type"));
        Assert.Null(openAIJson["tool_choice"]!["name"]);

        var claudeTool = ProviderTools.WebSearch();
        claudeTool.Id = "search_current_web";
        var claude = Request("claude-opus-4-8", claudeTool);
        claude.ToolChoice = new ToolChoice { Type = ToolChoiceType.Specific, ToolName = "search_current_web" };
        var claudeJson = Serialize(new ClaudeConverter().ConvertRequest(claude));
        Assert.Equal("web_search", Text(claudeJson["tool_choice"]!, "name"));

        var geminiTool = ProviderTools.WebSearch();
        geminiTool.Id = "search_current_web";
        var gemini = Request("gemini-3.6-flash", geminiTool);
        gemini.ToolChoice = new ToolChoice { Type = ToolChoiceType.Specific, ToolName = "search_current_web" };
        var geminiJson = Serialize(new GeminiInteractionsConverter().ConvertRequest(gemini));
        Assert.Equal("google_search",
            geminiJson["generation_config"]!["tool_choice"]!["allowed_tools"]!["tools"]![0]!.GetValue<string>());
    }

    [Fact]
    public async Task GeminiTimeRangeUsesGenerateContentCompatibilityPath()
    {
        const string response = """
        {"responseId":"gem_1","modelVersion":"gemini-3.6-flash","candidates":[{"index":0,"finishReason":"STOP","content":{"role":"model","parts":[{"text":"Hi"}]}}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}
        """;
        var handler = new TestHttpMessageHandler(response);
        using var client = new HttpClient(handler)
        {
            BaseAddress = new Uri("https://generativelanguage.googleapis.com/v1beta/")
        };
        var service = new GeminiService(client, "key");
        var request = Request("gemini-3.6-flash", ProviderTools.WebSearch(new WebSearchOptions
        {
            Gemini = new GeminiWebSearchOptions
            {
                StartTime = DateTimeOffset.Parse("2026-07-01T00:00:00Z"),
                EndTime = DateTimeOffset.Parse("2026-08-01T00:00:00Z")
            }
        }));

        await service.GenerateAsync(request);

        Assert.EndsWith(":generateContent", handler.LastRequest!.RequestUri!.AbsoluteUri);
        var json = JsonNode.Parse(handler.LastRequestBody!)!;
        Assert.StartsWith("2026-07-01T00:00:00", json["tools"]![0]!["googleSearch"]!["timeRangeFilter"]!["startTime"]!.GetValue<string>());
    }

    [Fact]
    public void UnsupportedConstraintsAndInvalidCombinationsFailValidation()
    {
        var gemini = Request("gemini-3.6-flash", ProviderTools.WebSearch(new WebSearchOptions
        {
            AllowedDomains = new[] { "example.com" }
        }));
        Assert.Contains(
            LLMRequestValidator.Validate(gemini, LLMProvider.Gemini),
            diagnostic => diagnostic.Code == "gemini.web_search.allowed_domains_unsupported" &&
                diagnostic.Severity == RequestDiagnosticSeverity.Error);

        var claude = Request("claude-opus-4-8", ProviderTools.WebSearch(new WebSearchOptions
        {
            AllowedDomains = new[] { "example.com" },
            BlockedDomains = new[] { "other.example" }
        }));
        Assert.Contains(
            LLMRequestValidator.Validate(claude, LLMProvider.Claude),
            diagnostic => diagnostic.Code == "claude.web_search.domain_filters_conflict");

        var badRange = Request("gemini-3.6-flash", ProviderTools.WebSearch(new WebSearchOptions
        {
            Gemini = new GeminiWebSearchOptions
            {
                StartTime = DateTimeOffset.Parse("2026-08-01T00:00:00Z")
            }
        }));
        Assert.Contains(
            LLMRequestValidator.Validate(badRange, LLMProvider.Gemini),
            diagnostic => diagnostic.Code == "gemini.web_search.time_range_pair");

        var forcedRange = Request("gemini-3.6-flash", ProviderTools.WebSearch(new WebSearchOptions
        {
            Gemini = new GeminiWebSearchOptions
            {
                StartTime = DateTimeOffset.Parse("2026-07-01T00:00:00Z"),
                EndTime = DateTimeOffset.Parse("2026-08-01T00:00:00Z")
            }
        }));
        forcedRange.ToolChoice = new ToolChoice { Type = ToolChoiceType.Required };
        Assert.Contains(
            LLMRequestValidator.Validate(forcedRange, LLMProvider.Gemini),
            diagnostic => diagnostic.Code == "gemini.web_search.time_range_tool_choice_unsupported");
    }

    private static UnifiedRequest Request(string model, LLMTool tool) => new()
    {
        Model = model,
        Messages = { new UnifiedMessage(MessageRole.User, "Search") },
        Tools = new ToolCollection { tool }
    };

    private static JsonObject Serialize(object value) =>
        JsonSerializer.SerializeToNode(value, new JsonSerializerOptions
        {
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        })!.AsObject();

    private static string Text(JsonNode node, string property) => node[property]!.GetValue<string>();
}
