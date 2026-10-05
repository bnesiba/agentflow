using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// Base type for the two supported tool definition kinds: application-defined
    /// functions and provider-native capabilities.
    /// </summary>
    public abstract class LLMTool
    {
        /// <summary>
        /// Stable request-local identifier used to target and correlate the tool.
        /// Function tools default to their function name; provider tools are assigned
        /// a stable capability identifier by <see cref="ProviderTools"/>.
        /// </summary>
        public string Id { get; set; } = string.Empty;
        public PromptCacheDirective? Cache { get; set; }
    }

    /// <summary>
    /// Application-defined JSON-schema function tool.
    /// </summary>
    public class FunctionTool : LLMTool
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public Dictionary<string, object> Parameters { get; set; } = new();
        public bool? Strict { get; set; }
    }

    /// <summary>
    /// Backward-compatible name for an application-defined function tool.
    /// </summary>
    public class ToolDefinition : FunctionTool
    {
    }

    /// <summary>
    /// A provider-native capability that is translated by the selected provider adapter.
    /// </summary>
    public sealed class ProviderTool : LLMTool
    {
        public ProviderToolCapability Capability { get; init; }
        public ProviderToolOptions Options { get; init; } = new EmptyProviderToolOptions();
    }

    public enum ProviderToolCapability
    {
        WebSearch
    }

    public abstract class ProviderToolOptions
    {
    }

    internal sealed class EmptyProviderToolOptions : ProviderToolOptions
    {
    }

    /// <summary>
    /// Collection shared by function tools and provider-native tools.
    /// </summary>
    public sealed class ToolCollection : Collection<LLMTool>
    {
        public ToolCollection()
        {
        }

        public ToolCollection(IEnumerable<LLMTool> tools)
            : base(tools.ToList())
        {
        }

        /// <summary>
        /// Preserves the common existing assignment pattern:
        /// request.Tools = new List&lt;ToolDefinition&gt; { ... }.
        /// </summary>
        public static implicit operator ToolCollection(List<ToolDefinition> tools) => new(tools);
    }

    [Flags]
    public enum WebSearchContentTypes
    {
        Web = 1,
        Image = 2
    }

    public sealed class ApproximateLocation
    {
        public string? City { get; init; }
        public string? Region { get; init; }
        public string? Country { get; init; }
        public string? Timezone { get; init; }
    }

    public sealed class WebSearchOptions : ProviderToolOptions
    {
        public IReadOnlyList<string>? AllowedDomains { get; init; }
        public IReadOnlyList<string>? BlockedDomains { get; init; }
        public ApproximateLocation? Location { get; init; }
        public WebSearchContentTypes ContentTypes { get; init; } = WebSearchContentTypes.Web;

        public OpenAIWebSearchOptions? OpenAI { get; init; }
        public AnthropicWebSearchOptions? Anthropic { get; init; }
        public GeminiWebSearchOptions? Gemini { get; init; }
    }

    public enum WebSearchContextSize
    {
        Low,
        Medium,
        High
    }

    public sealed class OpenAIWebSearchOptions
    {
        public WebSearchContextSize? ContextSize { get; init; }
        public bool IncludeAllSources { get; init; }
        public bool UnlimitedReturnTokenBudget { get; init; }
        public bool ExternalWebAccess { get; init; } = true;
        public int? MaximumImageResults { get; init; }
        public bool IncludeImageCaptions { get; init; }
    }

    public sealed class AnthropicWebSearchOptions
    {
        public int? MaximumUses { get; init; }
        public bool? DynamicFiltering { get; init; }
        public bool? IncludeFullResults { get; init; }
    }

    public sealed class GeminiWebSearchOptions
    {
        public DateTimeOffset? StartTime { get; init; }
        public DateTimeOffset? EndTime { get; init; }
    }

    public static class ProviderTools
    {
        public static ProviderTool WebSearch(WebSearchOptions? options = null) => new()
        {
            Id = "web_search",
            Capability = ProviderToolCapability.WebSearch,
            Options = options ?? new WebSearchOptions()
        };
    }
}
