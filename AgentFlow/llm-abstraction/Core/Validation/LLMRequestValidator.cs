using System;
using System.Collections.Generic;
using System.Linq;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Core.Validation
{
    public enum RequestDiagnosticSeverity
    {
        Information,
        Warning,
        Error
    }

    public sealed class RequestDiagnostic
    {
        public RequestDiagnosticSeverity Severity { get; init; }
        public string Code { get; init; } = string.Empty;
        public string Message { get; init; } = string.Empty;
        public string? PropertyPath { get; init; }
    }

    public sealed class LLMRequestValidationException : ArgumentException
    {
        public LLMRequestValidationException(
            LLMProvider provider,
            IReadOnlyList<RequestDiagnostic> diagnostics)
            : base(string.Join(Environment.NewLine, diagnostics.Select(diagnostic => diagnostic.Message)))
        {
            Provider = provider;
            Diagnostics = diagnostics;
        }

        public LLMProvider Provider { get; }
        public IReadOnlyList<RequestDiagnostic> Diagnostics { get; }
    }

    public static class LLMRequestValidator
    {
        public static ModelCapabilityProfile GetCapabilities(
            LLMProvider provider,
            string model,
            LLMApiSurface? surface = null) =>
            ModelCapabilityRegistry.Resolve(provider, model, surface);

        public static IReadOnlyList<RequestDiagnostic> Validate(
            UnifiedRequest request,
            LLMProvider provider,
            LLMApiSurface? surface = null)
        {
            ArgumentNullException.ThrowIfNull(request);
            var diagnostics = new List<RequestDiagnostic>();
            var capabilities = GetCapabilities(provider, request.Model, surface);

            AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(request.Model),
                "request.model.required", "A provider model identifier is required.", "Model");
            AddErrorIf(diagnostics,
                !request.Messages.Any(message => message.Role != MessageRole.System),
                "request.messages.required",
                "At least one non-system conversation message is required.",
                "Messages");
            AddErrorIf(diagnostics, request.Parameters.MaxOutputTokens < 0,
                "request.max_output_tokens.range",
                "MaxOutputTokens cannot be negative.",
                "Parameters.MaxOutputTokens");

            if (request.Parameters.Temperature is double temperature)
            {
                AddErrorIf(diagnostics,
                    temperature < capabilities.MinimumTemperature || temperature > capabilities.MaximumTemperature,
                    "request.temperature.range",
                    $"{provider} temperature must be between {capabilities.MinimumTemperature} and {capabilities.MaximumTemperature}.",
                    "Parameters.Temperature");
            }

            if (request.Parameters.TopP is double topP)
            {
                AddErrorIf(diagnostics, topP < 0 || topP > 1,
                    "request.top_p.range", "TopP must be between 0 and 1.", "Parameters.TopP");
            }

            AddErrorIf(diagnostics, request.Parameters.TopK <= 0,
                "request.top_k.range", "TopK must be greater than zero.", "Parameters.TopK");

            ValidateTools(request, diagnostics);
            ValidateProviderTools(request, provider, diagnostics);
            ValidateOutputFormat(request, capabilities, diagnostics);
            ValidateReasoning(request, capabilities, diagnostics);
            ValidatePromptCaching(request, provider, capabilities, diagnostics);
            ValidateProviderRules(request, provider, capabilities, diagnostics);

            if (request.Continuation != null && request.Continuation.Provider != ProviderName(provider))
            {
                diagnostics.Add(new RequestDiagnostic
                {
                    Severity = RequestDiagnosticSeverity.Warning,
                    Code = "request.continuation.provider_mismatch",
                    Message = $"Continuation state for '{request.Continuation.Provider}' will not be sent to {provider}.",
                    PropertyPath = "Continuation.Provider"
                });
            }

            return diagnostics;
        }

        private static void ValidatePromptCaching(
            UnifiedRequest request,
            LLMProvider provider,
            ModelCapabilityProfile capabilities,
            List<RequestDiagnostic> diagnostics)
        {
            var cache = request.Cache;
            var directives = EnumerateCacheDirectives(request).ToList();
            if (cache == null && directives.Count == 0)
                return;

            AddWarningIf(diagnostics, capabilities.Recognition == ModelRecognition.Unknown,
                "request.cache.unknown_model",
                $"Prompt-caching support has not been verified for model '{request.Model}' on {capabilities.Surface}; cache fields will still be sent.",
                "Cache");

            AddErrorIf(diagnostics, cache?.OpenAI != null && provider != LLMProvider.OpenAI,
                "request.cache.provider_options_mismatch",
                "OpenAI prompt-cache options can only be used with OpenAI.", "Cache.OpenAI");
            AddErrorIf(diagnostics, cache?.Gemini != null && provider != LLMProvider.Gemini,
                "request.cache.provider_options_mismatch",
                "Gemini prompt-cache options can only be used with Gemini.", "Cache.Gemini");
            AddErrorIf(diagnostics, cache?.Mode == PromptCacheMode.Disabled && directives.Count > 0,
                "request.cache.disabled_with_breakpoints",
                "Prompt-cache breakpoints cannot be supplied when caching is Disabled.", "Cache.Mode");

            switch (provider)
            {
                case LLMProvider.Claude:
                    ValidateAnthropicPromptCaching(request, directives, diagnostics);
                    break;
                case LLMProvider.OpenAI:
                    ValidateOpenAIPromptCaching(request, directives, diagnostics);
                    break;
                case LLMProvider.Gemini:
                    ValidateGeminiPromptCaching(request, capabilities.Surface, directives, diagnostics);
                    break;
            }
        }

        private static void ValidateAnthropicPromptCaching(
            UnifiedRequest request,
            List<(PromptCacheDirective Directive, string Path, ContentBlock? Block)> directives,
            List<RequestDiagnostic> diagnostics)
        {
            AddErrorIf(diagnostics, request.Cache?.OpenAI != null || request.Cache?.Gemini != null,
                "claude.cache.provider_options_unsupported",
                "Anthropic requests cannot use OpenAI or Gemini prompt-cache extensions.", "Cache");
            AddErrorIf(diagnostics,
                request.Cache?.Ttl is not null and not (PromptCacheTtl.FiveMinutes or PromptCacheTtl.OneHour),
                "claude.cache.ttl_unsupported",
                "Anthropic automatic prompt caching accepts only FiveMinutes or OneHour TTL.", "Cache.Ttl");
            var automaticBreakpointCount = request.Cache?.Mode == PromptCacheMode.PreferReuse ? 1 : 0;
            AddErrorIf(diagnostics, directives.Count + automaticBreakpointCount > 4,
                "claude.cache.breakpoint_limit",
                "Anthropic Messages accepts at most four cache breakpoints; automatic caching consumes one.", "Messages");

            var sawFiveMinute = false;
            foreach (var (directive, path, block) in directives)
            {
                AddErrorIf(diagnostics,
                    directive.Ttl is not null and not (PromptCacheTtl.FiveMinutes or PromptCacheTtl.OneHour),
                    "claude.cache.breakpoint_ttl_unsupported",
                    "Anthropic cache breakpoints accept only FiveMinutes or OneHour TTL.", path);
                AddErrorIf(diagnostics, block is ProviderNativeContent,
                    "claude.cache.native_block_unrepresentable",
                    "A cache breakpoint cannot be added to an opaque provider-native content block.", path);

                if (directive.Ttl is null or PromptCacheTtl.FiveMinutes)
                    sawFiveMinute = true;
                AddErrorIf(diagnostics, sawFiveMinute && directive.Ttl == PromptCacheTtl.OneHour,
                    "claude.cache.ttl_order",
                    "Anthropic requires one-hour cache prefixes to precede five-minute prefixes in tools, instructions, then message order.", path);
            }
        }

        private static void ValidateOpenAIPromptCaching(
            UnifiedRequest request,
            List<(PromptCacheDirective Directive, string Path, ContentBlock? Block)> directives,
            List<RequestDiagnostic> diagnostics)
        {
            AddErrorIf(diagnostics, request.Cache?.Gemini != null,
                "openai.cache.provider_options_unsupported",
                "OpenAI requests cannot use Gemini prompt-cache extensions.", "Cache.Gemini");
            AddErrorIf(diagnostics, request.Cache?.Ttl is not null and not PromptCacheTtl.ThirtyMinutes,
                "openai.cache.ttl_unsupported",
                "OpenAI prompt_cache_options accepts only the ThirtyMinutes TTL.", "Cache.Ttl");
            AddErrorIf(diagnostics, request.InstructionsCache != null,
                "openai.cache.instructions_breakpoint_unrepresentable",
                "OpenAI Responses explicit breakpoints can be placed on input text, image, or file items, not the top-level instructions field.",
                "InstructionsCache");
            AddErrorIf(diagnostics, request.Tools?.Any(tool => tool.Cache != null) == true,
                "openai.cache.tool_breakpoint_unrepresentable",
                "OpenAI Responses explicit breakpoints cannot be placed on tool declarations.", "Tools.Cache");
            AddErrorIf(diagnostics,
                request.Cache?.OpenAI?.Retention != null && request.Cache?.Ttl != null,
                "openai.cache.retention_ttl_conflict",
                "OpenAI legacy prompt_cache_retention cannot be combined with prompt_cache_options TTL.", "Cache");

            var hasExplicitBreakpoints = directives.Any(item => item.Block != null);
            AddWarningIf(diagnostics,
                hasExplicitBreakpoints && !request.Model.StartsWith("gpt-5.6", StringComparison.OrdinalIgnoreCase),
                "openai.cache.explicit_breakpoint_model_unverified",
                $"Explicit prompt-cache breakpoints are not verified for model '{request.Model}'; the fields will still be sent.",
                "Messages");
            AddWarningIf(diagnostics,
                request.Cache?.OpenAI?.Retention != null && request.Model.StartsWith("gpt-5.6", StringComparison.OrdinalIgnoreCase),
                "openai.cache.retention_model_unverified",
                $"Legacy prompt_cache_retention is not documented for model '{request.Model}'; the field will still be sent.",
                "Cache.OpenAI.Retention");

            foreach (var (directive, path, block) in directives)
            {
                AddErrorIf(diagnostics, directive.Ttl != null,
                    "openai.cache.breakpoint_ttl_unrepresentable",
                    "OpenAI TTL is request-wide; individual explicit breakpoints cannot specify a TTL.", path);
                AddErrorIf(diagnostics,
                    block != null && block is not TextContent && block is not ImageContent && block is not MediaContent,
                    "openai.cache.breakpoint_unrepresentable",
                    "OpenAI Responses explicit breakpoints can be placed only on input text, image, or file content.", path);
            }
        }

        private static void ValidateGeminiPromptCaching(
            UnifiedRequest request,
            LLMApiSurface surface,
            List<(PromptCacheDirective Directive, string Path, ContentBlock? Block)> directives,
            List<RequestDiagnostic> diagnostics)
        {
            AddErrorIf(diagnostics, request.Cache?.OpenAI != null,
                "gemini.cache.provider_options_unsupported",
                "Gemini requests cannot use OpenAI prompt-cache extensions.", "Cache.OpenAI");
            AddErrorIf(diagnostics, directives.Count > 0,
                "gemini.cache.breakpoints_unrepresentable",
                "Gemini does not expose per-block cache breakpoints; use implicit caching or a cachedContent resource.", "Messages");
            AddErrorIf(diagnostics, request.Cache?.Ttl != null,
                "gemini.cache.request_ttl_unrepresentable",
                "Gemini cache TTL is configured when a cachedContent resource is created, not on generation requests.", "Cache.Ttl");
            AddErrorIf(diagnostics, request.Cache?.Mode is PromptCacheMode.ExplicitBreakpointsOnly or PromptCacheMode.Disabled,
                "gemini.cache.mode_unrepresentable",
                "Gemini generation requests cannot select explicit-breakpoint-only or disabled implicit caching.", "Cache.Mode");
            AddErrorIf(diagnostics,
                surface == LLMApiSurface.GeminiInteractions &&
                !string.IsNullOrWhiteSpace(request.Cache?.Gemini?.CachedContentName),
                "gemini.interactions.explicit_cache_unrepresentable",
                "Gemini Interactions cannot reference cachedContent resources; use generateContent.", "Cache.Gemini.CachedContentName");
        }

        private static IEnumerable<(PromptCacheDirective Directive, string Path, ContentBlock? Block)>
            EnumerateCacheDirectives(UnifiedRequest request)
        {
            if (request.Tools != null)
            {
                for (var index = 0; index < request.Tools.Count; index++)
                    if (request.Tools[index].Cache is { } cache)
                        yield return (cache, $"Tools[{index}].Cache", null);
            }
            if (request.InstructionsCache != null)
                yield return (request.InstructionsCache, "InstructionsCache", null);
            for (var messageIndex = 0; messageIndex < request.Messages.Count; messageIndex++)
            {
                var content = request.Messages[messageIndex].Content;
                for (var contentIndex = 0; contentIndex < content.Count; contentIndex++)
                    if (content[contentIndex].Cache is { } cache)
                        yield return (cache, $"Messages[{messageIndex}].Content[{contentIndex}].Cache", content[contentIndex]);
            }
        }

        public static void ValidateAndThrow(
            UnifiedRequest request,
            LLMProvider provider,
            LLMApiSurface? surface = null)
        {
            var errors = Validate(request, provider, surface)
                .Where(diagnostic => diagnostic.Severity == RequestDiagnosticSeverity.Error)
                .ToList();
            if (errors.Count > 0)
                throw new LLMRequestValidationException(provider, errors);
        }

        private static void ValidateTools(
            UnifiedRequest request,
            List<RequestDiagnostic> diagnostics)
        {
            var tools = request.Tools ?? new ToolCollection();
            var functionTools = tools.OfType<FunctionTool>().ToList();
            for (var index = 0; index < functionTools.Count; index++)
            {
                AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(functionTools[index].Name),
                    "request.tool.name_required", "Every tool requires a name.", $"Tools[{index}].Name");
                AddErrorIf(diagnostics, functionTools[index].Parameters.Count == 0,
                    "request.tool.schema_required", $"Tool '{functionTools[index].Name}' requires a JSON input schema.", $"Tools[{index}].Parameters");
            }

            foreach (var duplicate in functionTools
                .Where(tool => !string.IsNullOrWhiteSpace(tool.Name))
                .GroupBy(tool => tool.Name, StringComparer.Ordinal)
                .Where(group => group.Count() > 1))
            {
                diagnostics.Add(Error(
                    "request.tool.name_duplicate",
                    $"Tool name '{duplicate.Key}' is defined more than once.",
                    "Tools"));
            }

            foreach (var providerTool in tools.OfType<ProviderTool>())
            {
                AddErrorIf(diagnostics,
                    functionTools.Any(function => function.Name == providerTool.Id),
                    "request.tool.identity_duplicate",
                    $"Tool identity '{providerTool.Id}' is shared by a function tool and a provider-native tool.",
                    "Tools");
            }

            if (request.ToolChoice?.Type == ToolChoiceType.Specific)
            {
                AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(request.ToolChoice.ToolName),
                    "request.tool_choice.name_required",
                    "Specific tool choice requires ToolName.",
                    "ToolChoice.ToolName");
                AddErrorIf(diagnostics,
                    !string.IsNullOrWhiteSpace(request.ToolChoice.ToolName) &&
                    functionTools.All(tool => tool.Name != request.ToolChoice.ToolName) &&
                    tools.All(tool => tool.Id != request.ToolChoice.ToolName),
                    "request.tool_choice.name_unknown",
                    $"Specific tool '{request.ToolChoice.ToolName}' is not present in Tools.",
                    "ToolChoice.ToolName");
            }
        }

        private static void ValidateOutputFormat(
            UnifiedRequest request,
            ModelCapabilityProfile capabilities,
            List<RequestDiagnostic> diagnostics)
        {
            if (request.Output == null)
                return;

            if (request.Output.Kind == OutputFormatKind.Text)
            {
                AddErrorIf(diagnostics, request.Output.JsonSchema != null,
                    "request.output.schema_unexpected",
                    "JsonSchema must be omitted when Output.Kind is Text.",
                    "Output.JsonSchema");
                return;
            }

            var support = request.Output.Kind == OutputFormatKind.JsonSchema
                ? capabilities.StructuredOutput
                : capabilities.JsonObjectOutput;
            AddCapabilityDiagnostic(
                diagnostics,
                support,
                "request.output.unsupported",
                "request.output.unknown_model",
                $"{capabilities.Model} does not support {request.Output.Kind} output on {capabilities.Surface}.",
                $"The abstraction has no verified {request.Output.Kind} capability data for model '{capabilities.Model}' on {capabilities.Surface}.",
                "Output.Kind");

            AddErrorIf(diagnostics,
                capabilities.Provider == LLMProvider.Claude &&
                request.Output.Kind == OutputFormatKind.JsonObject,
                "claude.output.json_object_unsupported",
                "Anthropic Messages has no unconstrained JSON-object output mode. Use JsonSchema with an explicit schema.",
                "Output.Kind");

            if (request.Output.Kind != OutputFormatKind.JsonSchema)
            {
                AddErrorIf(diagnostics, request.Output.JsonSchema != null,
                    "request.output.schema_unexpected",
                    "JsonSchema must be omitted unless Output.Kind is JsonSchema.",
                    "Output.JsonSchema");
                return;
            }

            AddErrorIf(diagnostics, request.Output.JsonSchema == null,
                "request.response_schema.required",
                "JsonSchema response format requires a schema definition.",
                "Output.JsonSchema");
            if (request.Output.JsonSchema != null)
            {
                AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(request.Output.JsonSchema.Name),
                    "request.response_schema.name_required",
                    "Response JSON Schema requires a stable name.",
                    "Output.JsonSchema.Name");
                AddErrorIf(diagnostics, request.Output.JsonSchema.Schema.Count == 0,
                    "request.response_schema.empty",
                    "Response JSON Schema cannot be empty.",
                    "Output.JsonSchema.Schema");
                if (request.Output.JsonSchema.Schema.Count > 0)
                    diagnostics.AddRange(JsonSchemaAnalyzer.Analyze(request.Output.JsonSchema));
            }
        }

        private static void ValidateReasoning(
            UnifiedRequest request,
            ModelCapabilityProfile capabilities,
            List<RequestDiagnostic> diagnostics)
        {
            var reasoning = request.Reasoning;
            if (reasoning == null)
                return;

            AddCapabilityDiagnostic(
                diagnostics,
                capabilities.Reasoning,
                "request.reasoning.unsupported",
                "request.reasoning.unknown_model",
                $"{capabilities.Model} does not support reasoning controls on {capabilities.Surface}.",
                $"The abstraction has no verified reasoning capability data for model '{capabilities.Model}' on {capabilities.Surface}.",
                "Reasoning");

            if (reasoning.Effort is ReasoningEffort effort &&
                capabilities.Recognition == ModelRecognition.Known &&
                !capabilities.ReasoningEfforts.Contains(effort))
            {
                diagnostics.Add(Warning(
                    "request.reasoning.effort_unsupported",
                    $"{capabilities.Model} is not documented to support reasoning effort {effort} on {capabilities.Surface}; the value will still be sent.",
                    "Reasoning.Effort"));
            }

            AddErrorIf(diagnostics,
                capabilities.Provider != LLMProvider.Claude && reasoning.Anthropic != null,
                "request.reasoning.provider_options_mismatch",
                "Anthropic reasoning options can only be used with Anthropic.",
                "Reasoning.Anthropic");
            AddErrorIf(diagnostics,
                capabilities.Provider != LLMProvider.OpenAI && reasoning.OpenAI != null,
                "request.reasoning.provider_options_mismatch",
                "OpenAI reasoning options can only be used with OpenAI.",
                "Reasoning.OpenAI");
            AddErrorIf(diagnostics,
                capabilities.Provider != LLMProvider.Gemini && reasoning.Gemini != null,
                "request.reasoning.provider_options_mismatch",
                "Gemini reasoning options can only be used with Gemini.",
                "Reasoning.Gemini");

            ValidateAnthropicReasoning(request, capabilities, diagnostics);
            ValidateOpenAIReasoning(reasoning, capabilities, diagnostics);
            ValidateGeminiReasoning(reasoning, capabilities, diagnostics);
        }

        private static void ValidateAnthropicReasoning(
            UnifiedRequest request,
            ModelCapabilityProfile capabilities,
            List<RequestDiagnostic> diagnostics)
        {
            if (capabilities.Provider != LLMProvider.Claude)
                return;

            if (request.Reasoning?.Anthropic == null)
            {
                AddErrorIf(diagnostics, request.Reasoning?.Output != null,
                    "claude.thinking.display_requires_mode",
                    "Anthropic reasoning output visibility requires explicit Anthropic thinking options.",
                    "Reasoning.Output");
                return;
            }

            var options = request.Reasoning.Anthropic;
            var mode = options.Mode;
            AddErrorIf(diagnostics,
                request.Reasoning.Output != null &&
                mode is AnthropicThinkingMode.Default or AnthropicThinkingMode.Disabled,
                "claude.thinking.display_requires_thinking",
                "Anthropic reasoning output visibility requires Adaptive or Manual thinking mode.",
                "Reasoning.Output");
            if (mode == AnthropicThinkingMode.Adaptive)
            {
                AddCapabilityDiagnostic(diagnostics, capabilities.AnthropicAdaptiveThinking,
                    "claude.thinking.adaptive_unsupported", "claude.thinking.adaptive_unknown_model",
                    $"{capabilities.Model} does not support adaptive thinking.",
                    $"Adaptive thinking support is not verified for model '{capabilities.Model}'.",
                    "Reasoning.Anthropic.Mode");
            }
            if (mode == AnthropicThinkingMode.Manual)
            {
                AddCapabilityDiagnostic(diagnostics, capabilities.AnthropicManualThinking,
                    "claude.thinking.manual_unsupported", "claude.thinking.manual_unknown_model",
                    $"{capabilities.Model} does not support manual thinking budgets.",
                    $"Manual thinking support is not verified for model '{capabilities.Model}'.",
                    "Reasoning.Anthropic.Mode");
                AddErrorIf(diagnostics, options.BudgetTokens == null,
                    "claude.thinking.budget_required",
                    "Manual Anthropic thinking requires BudgetTokens.",
                    "Reasoning.Anthropic.BudgetTokens");
            }
            else
            {
                AddErrorIf(diagnostics, options.BudgetTokens != null,
                    "claude.thinking.budget_mode",
                    "Anthropic BudgetTokens is valid only with Manual thinking mode.",
                    "Reasoning.Anthropic.BudgetTokens");
            }

            if (options.BudgetTokens is int budget)
            {
                AddErrorIf(diagnostics, budget < 1024,
                    "claude.thinking.budget_minimum",
                    "Claude manual thinking budget must be at least 1024 tokens.",
                    "Reasoning.Anthropic.BudgetTokens");
                AddErrorIf(diagnostics,
                    request.Parameters.MaxOutputTokens is int maximum && budget >= maximum,
                    "claude.thinking.budget_output_limit",
                    "Claude thinking budget must be lower than MaxOutputTokens.",
                    "Reasoning.Anthropic.BudgetTokens");
            }

            if (mode == AnthropicThinkingMode.Manual &&
                request.ToolChoice?.Type is ToolChoiceType.Required or ToolChoiceType.Specific)
            {
                diagnostics.Add(Error(
                    "claude.thinking.forced_tool_choice",
                    "Claude manual thinking supports only Auto or None tool choice.",
                    "ToolChoice.Type"));
            }

            if (mode is AnthropicThinkingMode.Manual or AnthropicThinkingMode.Adaptive)
            {
                AddErrorIf(diagnostics, request.Parameters.Temperature != null,
                    "claude.thinking.temperature_unsupported",
                    "Claude thinking does not accept an explicit Temperature setting.",
                    "Parameters.Temperature");
                AddErrorIf(diagnostics, request.Parameters.TopK != null,
                    "claude.thinking.top_k_unsupported",
                    "Claude thinking does not accept an explicit TopK setting.",
                    "Parameters.TopK");
                AddErrorIf(diagnostics,
                    request.Parameters.TopP is double topP && (topP < 0.95 || topP > 1),
                    "claude.thinking.top_p_range",
                    "Claude thinking accepts TopP only between 0.95 and 1.0.",
                    "Parameters.TopP");
            }

            AddWarningIf(diagnostics,
                capabilities.Family == "claude-5" &&
                capabilities.Model.StartsWith("claude-opus-5", StringComparison.OrdinalIgnoreCase) &&
                mode == AnthropicThinkingMode.Disabled &&
                request.Reasoning.Effort is ReasoningEffort.XHigh or ReasoningEffort.Max,
                "claude.thinking.disabled_effort_conflict",
                "Claude Opus 5 cannot disable thinking at XHigh or Max effort.",
                "Reasoning");
        }

        private static void ValidateOpenAIReasoning(
            ReasoningOptions reasoning,
            ModelCapabilityProfile capabilities,
            List<RequestDiagnostic> diagnostics)
        {
            if (capabilities.Provider != LLMProvider.OpenAI || reasoning.OpenAI == null)
                return;
            AddErrorIf(diagnostics,
                reasoning.Output == ReasoningOutput.Omitted && reasoning.OpenAI.Summary != null,
                "openai.reasoning.summary_conflict",
                "OpenAI reasoning Summary cannot be requested when portable reasoning output is Omitted.",
                "Reasoning");
            if (reasoning.OpenAI.Mode == OpenAIReasoningMode.Pro)
            {
                AddCapabilityDiagnostic(diagnostics, capabilities.OpenAIProMode,
                    "openai.reasoning.pro_unsupported", "openai.reasoning.pro_unknown_model",
                    $"{capabilities.Model} does not support reasoning mode Pro.",
                    $"Pro reasoning mode is not verified for model '{capabilities.Model}'.",
                    "Reasoning.OpenAI.Mode");
            }
            if (reasoning.OpenAI.Context != null)
            {
                AddCapabilityDiagnostic(diagnostics, capabilities.OpenAIReasoningContext,
                    "openai.reasoning.context_unsupported", "openai.reasoning.context_unknown_model",
                    $"{capabilities.Model} does not support persisted reasoning context controls.",
                    $"Reasoning context controls are not verified for model '{capabilities.Model}'.",
                    "Reasoning.OpenAI.Context");
            }
        }

        private static void ValidateGeminiReasoning(
            ReasoningOptions reasoning,
            ModelCapabilityProfile capabilities,
            List<RequestDiagnostic> diagnostics)
        {
            if (capabilities.Provider != LLMProvider.Gemini || reasoning.Gemini == null)
                return;
            if (reasoning.Gemini.ThinkingBudget is int budget)
            {
                AddCapabilityDiagnostic(diagnostics, capabilities.ThinkingBudget,
                    "gemini.thinking.budget_unsupported", "gemini.thinking.budget_unknown_model",
                    $"{capabilities.Model} does not support ThinkingBudget on {capabilities.Surface}.",
                    $"ThinkingBudget support is not verified for model '{capabilities.Model}' on {capabilities.Surface}.",
                    "Reasoning.Gemini.ThinkingBudget");
                AddErrorIf(diagnostics,
                    capabilities.Surface == LLMApiSurface.GeminiInteractions,
                    "gemini.interactions.thinking_budget_unrepresentable",
                    "Gemini Interactions has no ThinkingBudget field; use a portable Effort thinking level or generateContent.",
                    "Reasoning.Gemini.ThinkingBudget");
                AddErrorIf(diagnostics, budget < 0,
                    "gemini.thinking.budget_range",
                    "Gemini ThinkingBudget cannot be negative.",
                    "Reasoning.Gemini.ThinkingBudget");
                AddErrorIf(diagnostics, reasoning.Effort != null,
                    "gemini.thinking.conflicting_controls",
                    "Gemini ThinkingBudget and portable reasoning Effort cannot be selected together.",
                    "Reasoning");
            }
        }

        private static void ValidateProviderTools(
            UnifiedRequest request,
            LLMProvider provider,
            List<RequestDiagnostic> diagnostics)
        {
            var providerTools = request.Tools?.OfType<ProviderTool>().ToList() ?? new();
            foreach (var tool in providerTools)
            {
                AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(tool.Id),
                    "request.provider_tool.id_required",
                    "Every provider-native tool requires a stable ID.",
                    "Tools.Id");

                if (tool.Capability != ProviderToolCapability.WebSearch)
                    continue;

                if (tool.Options is not WebSearchOptions options)
                {
                    diagnostics.Add(Error(
                        "request.web_search.options_type",
                        "Web Search requires WebSearchOptions.",
                        "Tools.Options"));
                    continue;
                }

                AddErrorIf(diagnostics,
                    options.ContentTypes == 0 ||
                    (options.ContentTypes & ~(WebSearchContentTypes.Web | WebSearchContentTypes.Image)) != 0,
                    "request.web_search.content_types",
                    "Web Search requires at least one recognized content type.",
                    "Tools.Options.ContentTypes");

                ValidateDomains(options.AllowedDomains, "AllowedDomains", diagnostics);
                ValidateDomains(options.BlockedDomains, "BlockedDomains", diagnostics);

                if (options.Location != null && provider is LLMProvider.Claude or LLMProvider.OpenAI)
                {
                    AddErrorIf(diagnostics,
                        string.IsNullOrWhiteSpace(options.Location.City) &&
                        string.IsNullOrWhiteSpace(options.Location.Region) &&
                        string.IsNullOrWhiteSpace(options.Location.Country) &&
                        string.IsNullOrWhiteSpace(options.Location.Timezone),
                        "request.web_search.location_empty",
                        "Approximate Web Search location requires at least one location field.",
                        "Tools.Options.Location");
                    AddErrorIf(diagnostics,
                        options.Location.Country != null && options.Location.Country.Length != 2,
                        "request.web_search.location_country",
                        "Approximate Web Search country must be a two-letter ISO country code.",
                        "Tools.Options.Location.Country");
                }

                if (provider == LLMProvider.Claude)
                {
                    AddErrorIf(diagnostics,
                        options.AllowedDomains?.Count > 0 && options.BlockedDomains?.Count > 0,
                        "claude.web_search.domain_filters_conflict",
                        "Anthropic Web Search accepts allowed domains or blocked domains, not both.",
                        "Tools.Options");
                    AddErrorIf(diagnostics,
                        options.ContentTypes.HasFlag(WebSearchContentTypes.Image),
                        "claude.web_search.image_unsupported",
                        "Anthropic Web Search does not expose portable image search.",
                        "Tools.Options.ContentTypes");
                    AddErrorIf(diagnostics, options.Anthropic?.MaximumUses <= 0,
                        "claude.web_search.max_uses",
                        "Anthropic Web Search MaximumUses must be greater than zero.",
                        "Tools.Options.Anthropic.MaximumUses");
                }

                if (provider == LLMProvider.OpenAI)
                {
                    AddErrorIf(diagnostics,
                        options.AllowedDomains?.Any(domain => domain.Contains('/')) == true ||
                        options.BlockedDomains?.Any(domain => domain.Contains('/')) == true,
                        "openai.web_search.domain_path_unsupported",
                        "OpenAI Web Search filters accept bare domains without URL paths.",
                        "Tools.Options");
                    AddErrorIf(diagnostics,
                        (options.AllowedDomains?.Count ?? 0) > 100 ||
                        (options.BlockedDomains?.Count ?? 0) > 100,
                        "openai.web_search.domain_limit",
                        "OpenAI Web Search accepts at most 100 allowed or blocked domains.",
                        "Tools.Options");
                    AddErrorIf(diagnostics, options.OpenAI?.MaximumImageResults <= 0,
                        "openai.web_search.max_image_results",
                        "OpenAI MaximumImageResults must be greater than zero.",
                        "Tools.Options.OpenAI.MaximumImageResults");
                }

                if (provider == LLMProvider.Gemini)
                {
                    AddErrorIf(diagnostics, options.AllowedDomains?.Count > 0,
                        "gemini.web_search.allowed_domains_unsupported",
                        "Gemini Google Search cannot enforce allowed-domain restrictions.",
                        "Tools.Options.AllowedDomains");
                    AddErrorIf(diagnostics, options.BlockedDomains?.Count > 0,
                        "gemini.web_search.blocked_domains_unsupported",
                        "Gemini Google Search cannot enforce blocked-domain restrictions.",
                        "Tools.Options.BlockedDomains");
                    AddErrorIf(diagnostics, options.Location != null,
                        "gemini.web_search.location_unsupported",
                        "Gemini Google Search does not expose an approximate-location option.",
                        "Tools.Options.Location");

                    var start = options.Gemini?.StartTime;
                    var end = options.Gemini?.EndTime;
                    AddErrorIf(diagnostics, (start == null) != (end == null),
                        "gemini.web_search.time_range_pair",
                        "Gemini Web Search requires both StartTime and EndTime.",
                        "Tools.Options.Gemini");
                    AddErrorIf(diagnostics, start != null && end != null && start >= end,
                        "gemini.web_search.time_range_order",
                        "Gemini Web Search StartTime must be earlier than EndTime.",
                        "Tools.Options.Gemini");
                    AddErrorIf(diagnostics,
                        start != null && end != null &&
                        request.ToolChoice != null && request.ToolChoice.Type != ToolChoiceType.Auto,
                        "gemini.web_search.time_range_tool_choice_unsupported",
                        "Gemini's generateContent time-range compatibility path cannot faithfully enforce portable Web Search tool choice.",
                        "ToolChoice.Type");
                }
            }

            foreach (var duplicate in providerTools
                .Where(tool => !string.IsNullOrWhiteSpace(tool.Id))
                .GroupBy(tool => tool.Id, StringComparer.Ordinal)
                .Where(group => group.Count() > 1))
            {
                diagnostics.Add(Error(
                    "request.provider_tool.id_duplicate",
                    $"Provider-native tool ID '{duplicate.Key}' is defined more than once.",
                    "Tools"));
            }
        }

        private static void ValidateDomains(
            IReadOnlyList<string>? domains,
            string property,
            List<RequestDiagnostic> diagnostics)
        {
            if (domains == null)
                return;
            foreach (var domain in domains)
            {
                AddErrorIf(diagnostics,
                    string.IsNullOrWhiteSpace(domain) ||
                    domain.Contains("://", StringComparison.Ordinal),
                    "request.web_search.domain_format",
                    "Web Search domains must be non-empty and omit the URL scheme.",
                    $"Tools.Options.{property}");
            }
        }

        private static void ValidateProviderRules(
            UnifiedRequest request,
            LLMProvider provider,
            ModelCapabilityProfile capabilities,
            List<RequestDiagnostic> diagnostics)
        {
            var hasSampling = request.Parameters.Temperature != null ||
                request.Parameters.TopP != null || request.Parameters.TopK != null;
            if (hasSampling)
            {
                AddCapabilityDiagnostic(diagnostics, capabilities.SamplingControls,
                    "request.sampling.unsupported", "request.sampling.unknown_model",
                    $"{capabilities.Model} does not accept explicit sampling controls on {capabilities.Surface}.",
                    $"Sampling-control support is not verified for model '{capabilities.Model}' on {capabilities.Surface}.",
                    "Parameters");
            }

            if (provider == LLMProvider.OpenAI)
            {
                AddErrorIf(diagnostics, request.Parameters.TopK != null,
                    "openai.top_k.unsupported",
                    "OpenAI Responses does not expose TopK.",
                    "Parameters.TopK");
                AddErrorIf(diagnostics, request.Parameters.StopSequences?.Count > 0,
                    "openai.stop_sequences.unsupported",
                    "OpenAI Responses does not expose stop sequences.",
                    "Parameters.StopSequences");
                ValidateNonAnthropicCitationInputs(request, "OpenAI Responses", diagnostics);
            }

            if (provider == LLMProvider.Claude)
            {
                AddErrorIf(diagnostics, request.ToolChoice?.DisableParallelToolUse != null,
                    "claude.parallel_tool_control.unmapped",
                    "Claude DisableParallelToolUse is not currently mapped by the converter.",
                    "ToolChoice.DisableParallelToolUse");
                AddErrorIf(diagnostics, request.Metadata?.Tags?.Count > 0,
                    "claude.metadata_tags.unsupported",
                    "Claude request metadata does not accept portable tags; only UserId is supported.",
                    "Metadata.Tags");
                AddErrorIf(diagnostics,
                    request.Output?.Kind == OutputFormatKind.JsonSchema &&
                    (request.Messages.SelectMany(message => message.Content)
                        .OfType<DocumentContent>().Any(document => document.CitationsEnabled == true) ||
                     request.Messages.SelectMany(message => message.Content)
                        .OfType<SearchResultContent>().Any(result => result.CitationsEnabled == true)),
                    "claude.citations.structured_output_conflict",
                    "Anthropic document citations cannot be combined with structured output.",
                    "Output.Kind");
                foreach (var result in request.Messages.SelectMany(message => message.Content).OfType<SearchResultContent>())
                {
                    AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(result.Source),
                        "claude.search_result.source_required",
                        "Anthropic search-result input requires Source.", "Messages.Content.Source");
                    AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(result.Title),
                        "claude.search_result.title_required",
                        "Anthropic search-result input requires Title.", "Messages.Content.Title");
                    AddErrorIf(diagnostics, result.Content.Count == 0,
                        "claude.search_result.content_required",
                        "Anthropic search-result input requires at least one text block.", "Messages.Content.Content");
                    AddErrorIf(diagnostics, result.Content.Any(text => text.Cache != null),
                        "claude.search_result.inner_cache_unrepresentable",
                        "Cache the search-result block itself; nested search-result text cannot carry cache directives.",
                        "Messages.Content.Content.Cache");
                }
            }

            if (provider == LLMProvider.Gemini)
            {
                AddErrorIf(diagnostics,
                    capabilities.Surface == LLMApiSurface.GeminiInteractions &&
                    request.Parameters.Temperature != null,
                    "gemini.interactions.temperature_unrepresentable",
                    "Gemini Interactions does not expose Temperature; use generateContent.",
                    "Parameters.Temperature");
                AddErrorIf(diagnostics,
                    capabilities.Surface == LLMApiSurface.GeminiInteractions &&
                    request.Parameters.TopP != null,
                    "gemini.interactions.top_p_unrepresentable",
                    "Gemini Interactions does not expose TopP; use generateContent.",
                    "Parameters.TopP");
                AddErrorIf(diagnostics,
                    capabilities.Surface == LLMApiSurface.GeminiInteractions &&
                    request.Parameters.TopK != null,
                    "gemini.interactions.top_k_unrepresentable",
                    "Gemini Interactions does not expose TopK; use generateContent.",
                    "Parameters.TopK");
                AddErrorIf(diagnostics, request.ToolChoice?.DisableParallelToolUse != null,
                    "gemini.parallel_tool_control.unmapped",
                    "Gemini DisableParallelToolUse is not currently mapped by the converter.",
                    "ToolChoice.DisableParallelToolUse");
                AddErrorIf(diagnostics, request.Tools?.OfType<FunctionTool>().Any(tool => tool.Strict != null) == true,
                    "gemini.tool_strict.unmapped",
                    "Gemini function declarations do not map the portable Strict setting.",
                    "Tools.Strict");
                AddErrorIf(diagnostics,
                    request.Metadata?.UserId != null || request.Metadata?.Tags?.Count > 0,
                    "gemini.metadata.unsupported",
                    "Portable request metadata is not supported by the selected Gemini surface.",
                    "Metadata");
                ValidateNonAnthropicCitationInputs(request, "Gemini", diagnostics);
            }

            foreach (var message in request.Messages)
            {
                foreach (var result in message.Content.OfType<ToolResultContent>())
                {
                    AddErrorIf(diagnostics, string.IsNullOrWhiteSpace(result.ToolCallId),
                        "request.tool_result.call_id_required",
                        "Every tool result requires ToolCallId.",
                        "Messages.Content.ToolCallId");
                }
            }
        }

        private static void ValidateNonAnthropicCitationInputs(
            UnifiedRequest request,
            string providerName,
            List<RequestDiagnostic> diagnostics)
        {
            var blocks = request.Messages.SelectMany(message => message.Content).ToList();
            AddErrorIf(diagnostics, blocks.OfType<SearchResultContent>().Any(),
                "request.search_result_input.unrepresentable",
                $"{providerName} has no input block equivalent to Anthropic's citable search_result block.",
                "Messages.Content");
            foreach (var document in blocks.OfType<DocumentContent>())
            {
                AddErrorIf(diagnostics,
                    document.Title != null || document.Context != null || document.CitationsEnabled != null,
                    "request.document_citation_options.unrepresentable",
                    $"{providerName} can send the document, but cannot represent Anthropic title, context, or citation-enablement fields.",
                    "Messages.Content");
            }
        }

        private static string ProviderName(LLMProvider provider) => provider switch
        {
            LLMProvider.OpenAI => ProviderIds.OpenAI,
            LLMProvider.Claude => ProviderIds.Claude,
            LLMProvider.Gemini => ProviderIds.Gemini,
            _ => string.Empty
        };

        private static void AddErrorIf(
            List<RequestDiagnostic> diagnostics,
            bool condition,
            string code,
            string message,
            string? propertyPath)
        {
            if (condition)
                diagnostics.Add(Error(code, message, propertyPath));
        }

        private static void AddWarningIf(
            List<RequestDiagnostic> diagnostics,
            bool condition,
            string code,
            string message,
            string? propertyPath)
        {
            if (!condition)
                return;
            diagnostics.Add(new RequestDiagnostic
            {
                Severity = RequestDiagnosticSeverity.Warning,
                Code = code,
                Message = message,
                PropertyPath = propertyPath
            });
        }

        private static void AddCapabilityDiagnostic(
            List<RequestDiagnostic> diagnostics,
            CapabilitySupport support,
            string unsupportedCode,
            string unknownCode,
            string unsupportedMessage,
            string unknownMessage,
            string? propertyPath)
        {
            if (support == CapabilitySupport.Supported)
                return;
            diagnostics.Add(Warning(
                support == CapabilitySupport.Unsupported ? unsupportedCode : unknownCode,
                support == CapabilitySupport.Unsupported ? unsupportedMessage : unknownMessage,
                propertyPath));
        }

        private static RequestDiagnostic Error(string code, string message, string? propertyPath)
        {
            return new RequestDiagnostic
            {
                Severity = RequestDiagnosticSeverity.Error,
                Code = code,
                Message = message,
                PropertyPath = propertyPath
            };
        }

        private static RequestDiagnostic Warning(string code, string message, string? propertyPath)
        {
            return new RequestDiagnostic
            {
                Severity = RequestDiagnosticSeverity.Warning,
                Code = code,
                Message = message,
                PropertyPath = propertyPath
            };
        }
    }
}
