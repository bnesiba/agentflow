using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using LLMAbstraction.Core.Models;

namespace LLMAbstraction.Core.Validation
{
    /// <summary>
    /// Validates the deliberately small JSON Schema intersection guaranteed by
    /// all supported provider structured-output converters.
    /// </summary>
    public static class JsonSchemaAnalyzer
    {
        private const int MaximumDepth = 5;
        private const int MaximumProperties = 100;

        private static readonly HashSet<string> PortableKeywords = new(StringComparer.Ordinal)
        {
            "$defs", "$ref", "type", "properties", "required", "additionalProperties",
            "items", "enum", "anyOf", "description", "title", "format"
        };

        public static IReadOnlyList<RequestDiagnostic> Analyze(JsonSchemaDefinition definition)
        {
            ArgumentNullException.ThrowIfNull(definition);
            var diagnostics = new List<RequestDiagnostic>();
            JsonElement root;
            try
            {
                root = JsonSerializer.SerializeToElement(definition.Schema);
            }
            catch (Exception exception) when (exception is JsonException or NotSupportedException)
            {
                diagnostics.Add(Error(
                    "request.response_schema.not_json",
                    $"Response schema contains a value that cannot be represented as JSON: {exception.Message}",
                    "Output.JsonSchema.Schema"));
                return diagnostics;
            }

            if (root.ValueKind != JsonValueKind.Object)
            {
                diagnostics.Add(Error(
                    "request.response_schema.root_object",
                    "Portable structured output requires a JSON object schema at the root.",
                    "Output.JsonSchema.Schema"));
                return diagnostics;
            }

            if (!root.TryGetProperty("type", out var rootType) ||
                rootType.ValueKind != JsonValueKind.String ||
                rootType.GetString() != "object")
            {
                diagnostics.Add(Error(
                    "request.response_schema.root_type",
                    "Portable structured output requires type 'object' at the schema root.",
                    "Output.JsonSchema.Schema"));
            }
            if (root.TryGetProperty("anyOf", out _))
            {
                diagnostics.Add(Error(
                    "request.response_schema.root_anyof",
                    "Portable structured output does not allow anyOf at the schema root.",
                    "Output.JsonSchema.Schema"));
            }

            var propertyCount = 0;
            AnalyzeNode(root, "$", 1, diagnostics, ref propertyCount);
            if (propertyCount > MaximumProperties)
            {
                diagnostics.Add(Error(
                    "request.response_schema.property_limit",
                    $"Portable structured output supports at most {MaximumProperties} object properties; the schema contains {propertyCount}.",
                    "Output.JsonSchema.Schema"));
            }
            return diagnostics;
        }

        private static void AnalyzeNode(
            JsonElement node,
            string path,
            int depth,
            List<RequestDiagnostic> diagnostics,
            ref int propertyCount)
        {
            if (node.ValueKind != JsonValueKind.Object)
                return;

            if (depth > MaximumDepth)
            {
                diagnostics.Add(Error(
                    "request.response_schema.depth_limit",
                    $"Portable structured output supports at most {MaximumDepth} schema levels; '{path}' is deeper.",
                    "Output.JsonSchema.Schema"));
                return;
            }

            foreach (var property in node.EnumerateObject())
            {
                if (!PortableKeywords.Contains(property.Name))
                {
                    diagnostics.Add(Error(
                        "request.response_schema.keyword_unsupported",
                        $"JSON Schema keyword '{property.Name}' at '{path}' is outside the portable provider intersection.",
                        "Output.JsonSchema.Schema"));
                }
            }

            if (node.TryGetProperty("type", out var type) && !IsValidType(type))
            {
                diagnostics.Add(Error(
                    "request.response_schema.type_invalid",
                    $"Schema type at '{path}' must be a supported JSON Schema primitive or an array of supported primitives.",
                    "Output.JsonSchema.Schema"));
            }

            if (node.TryGetProperty("format", out var format) &&
                (format.ValueKind != JsonValueKind.String ||
                 format.GetString() is not ("date-time" or "date" or "time")))
            {
                diagnostics.Add(Error(
                    "request.response_schema.format_unsupported",
                    $"String format at '{path}' must be date-time, date, or time for portable structured output.",
                    "Output.JsonSchema.Schema"));
            }

            if (node.TryGetProperty("properties", out var properties))
            {
                if (properties.ValueKind != JsonValueKind.Object)
                {
                    diagnostics.Add(Error(
                        "request.response_schema.properties_object",
                        $"Schema properties at '{path}' must be an object.",
                        "Output.JsonSchema.Schema"));
                }
                else
                {
                    var names = properties.EnumerateObject().Select(property => property.Name).ToHashSet(StringComparer.Ordinal);
                    propertyCount += names.Count;
                    var required = ReadRequired(node, path, diagnostics);
                    var missing = names.Except(required, StringComparer.Ordinal).OrderBy(name => name).ToArray();
                    if (missing.Length > 0)
                    {
                        diagnostics.Add(Error(
                            "request.response_schema.all_properties_required",
                            $"Every object property must be listed in required for portable strict output. Missing at '{path}': {string.Join(", ", missing)}. Use a null union for optional values.",
                            "Output.JsonSchema.Schema"));
                    }

                    if (!node.TryGetProperty("additionalProperties", out var additional) ||
                        additional.ValueKind != JsonValueKind.False)
                    {
                        diagnostics.Add(Error(
                            "request.response_schema.additional_properties",
                            $"Object schema at '{path}' must set additionalProperties to false.",
                            "Output.JsonSchema.Schema"));
                    }

                    foreach (var property in properties.EnumerateObject())
                        AnalyzeNode(property.Value, $"{path}.properties.{property.Name}", depth + 1, diagnostics, ref propertyCount);
                }
            }

            if (node.TryGetProperty("items", out var items))
                AnalyzeNode(items, $"{path}.items", depth + 1, diagnostics, ref propertyCount);
            if (node.TryGetProperty("anyOf", out var anyOf) && anyOf.ValueKind == JsonValueKind.Array)
            {
                var index = 0;
                foreach (var item in anyOf.EnumerateArray())
                    AnalyzeNode(item, $"{path}.anyOf[{index++}]", depth + 1, diagnostics, ref propertyCount);
            }
            if (node.TryGetProperty("$defs", out var definitions) && definitions.ValueKind == JsonValueKind.Object)
            {
                foreach (var definition in definitions.EnumerateObject())
                    AnalyzeNode(definition.Value, $"{path}.$defs.{definition.Name}", depth + 1, diagnostics, ref propertyCount);
            }
        }

        private static HashSet<string> ReadRequired(
            JsonElement node,
            string path,
            List<RequestDiagnostic> diagnostics)
        {
            if (!node.TryGetProperty("required", out var required) || required.ValueKind != JsonValueKind.Array)
            {
                diagnostics.Add(Error(
                    "request.response_schema.required_array",
                    $"Object schema at '{path}' must provide a required array.",
                    "Output.JsonSchema.Schema"));
                return new HashSet<string>(StringComparer.Ordinal);
            }

            var result = new HashSet<string>(StringComparer.Ordinal);
            foreach (var item in required.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.String || item.GetString() is not string name)
                {
                    diagnostics.Add(Error(
                        "request.response_schema.required_string",
                        $"Every required entry at '{path}' must be a property name string.",
                        "Output.JsonSchema.Schema"));
                    continue;
                }
                result.Add(name);
            }
            return result;
        }

        private static bool IsValidType(JsonElement type)
        {
            if (type.ValueKind == JsonValueKind.String)
                return IsValidTypeName(type.GetString());
            return type.ValueKind == JsonValueKind.Array &&
                type.EnumerateArray().All(item =>
                    item.ValueKind == JsonValueKind.String && IsValidTypeName(item.GetString()));
        }

        private static bool IsValidTypeName(string? type) => type is
            "object" or "array" or "string" or "number" or "integer" or "boolean" or "null";

        private static RequestDiagnostic Error(string code, string message, string propertyPath) => new()
        {
            Severity = RequestDiagnosticSeverity.Error,
            Code = code,
            Message = message,
            PropertyPath = propertyPath
        };
    }
}
