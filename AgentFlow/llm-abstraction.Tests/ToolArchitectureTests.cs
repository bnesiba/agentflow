using LLMAbstraction.Core.Models;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Validation;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class ToolArchitectureTests
{
    [Fact]
    public void ExistingFunctionToolListAssignmentRemainsSupported()
    {
        var request = new UnifiedRequest
        {
            Tools = new List<ToolDefinition>
            {
                WeatherTool()
            }
        };

        var function = Assert.IsType<ToolDefinition>(Assert.Single(request.Tools!));
        Assert.Equal("get_weather", function.Name);
    }

    [Fact]
    public void FunctionAndProviderToolsShareOneCollection()
    {
        var request = new UnifiedRequest
        {
            Tools = new ToolCollection
            {
                WeatherTool(),
                ProviderTools.WebSearch()
            }
        };

        Assert.Collection(
            request.Tools!,
            tool => Assert.IsAssignableFrom<FunctionTool>(tool),
            tool =>
            {
                var providerTool = Assert.IsType<ProviderTool>(tool);
                Assert.Equal("web_search", providerTool.Id);
                Assert.Equal(ProviderToolCapability.WebSearch, providerTool.Capability);
                Assert.IsType<WebSearchOptions>(providerTool.Options);
            });
    }

    [Fact]
    public void WebSearchFactoryRetainsPortableAndNativeOptions()
    {
        var start = DateTimeOffset.Parse("2026-07-01T00:00:00Z");
        var end = DateTimeOffset.Parse("2026-08-01T00:00:00Z");

        var tool = ProviderTools.WebSearch(new WebSearchOptions
        {
            AllowedDomains = new[] { "example.com" },
            Location = new ApproximateLocation { Country = "US" },
            ContentTypes = WebSearchContentTypes.Web | WebSearchContentTypes.Image,
            Gemini = new GeminiWebSearchOptions { StartTime = start, EndTime = end }
        });

        var options = Assert.IsType<WebSearchOptions>(tool.Options);
        Assert.Equal("example.com", Assert.Single(options.AllowedDomains!));
        Assert.Equal("US", options.Location?.Country);
        Assert.True(options.ContentTypes.HasFlag(WebSearchContentTypes.Image));
        Assert.Equal(start, options.Gemini?.StartTime);
        Assert.Equal(end, options.Gemini?.EndTime);
    }

    [Fact]
    public void FunctionAndProviderToolIdentitiesCannotBeAmbiguous()
    {
        var function = WeatherTool();
        function.Name = "web_search";
        var request = new UnifiedRequest
        {
            Model = "gpt-5.6",
            Messages = { new UnifiedMessage(MessageRole.User, "Search") },
            Tools = new ToolCollection { function, ProviderTools.WebSearch() }
        };

        Assert.Contains(
            LLMRequestValidator.Validate(request, LLMProvider.OpenAI),
            diagnostic => diagnostic.Code == "request.tool.identity_duplicate");
    }

    private static ToolDefinition WeatherTool() => new()
    {
        Name = "get_weather",
        Description = "Gets weather.",
        Parameters = new Dictionary<string, object>
        {
            ["type"] = "object",
            ["properties"] = new Dictionary<string, object>()
        }
    };
}
