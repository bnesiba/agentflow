using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Validation;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.OpenAI.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class EvidenceInputTests
{
    [Fact]
    public void AnthropicSerializesCitableDocumentsAndSearchResults()
    {
        var document = new DocumentContent
        {
            MediaType = "application/pdf",
            Source = new MediaSource { FileId = "file_123" },
            Title = "Contract",
            Context = "Customer agreement",
            CitationsEnabled = true
        };
        var searchResult = new SearchResultContent
        {
            Source = "https://example.com/terms",
            Title = "Terms",
            Content = new List<TextContent> { new() { Text = "The current terms." } },
            CitationsEnabled = true,
            Cache = new PromptCacheDirective()
        };
        var request = Request("claude-sonnet-4-6", document, searchResult);

        var json = Serialize(new ClaudeConverter().ConvertRequest(request));
        var blocks = json["messages"]![0]!["content"]!;

        Assert.Equal("document", blocks[0]!["type"]!.GetValue<string>());
        Assert.Equal("file_123", blocks[0]!["source"]!["file_id"]!.GetValue<string>());
        Assert.True(blocks[0]!["citations"]!["enabled"]!.GetValue<bool>());
        Assert.Equal("search_result", blocks[1]!["type"]!.GetValue<string>());
        Assert.Equal("The current terms.", blocks[1]!["content"]![0]!["text"]!.GetValue<string>());
        Assert.NotNull(blocks[1]!["cache_control"]);
    }

    [Theory]
    [InlineData(LLMProvider.OpenAI, LLMApiSurface.OpenAIResponses)]
    [InlineData(LLMProvider.Gemini, LLMApiSurface.GeminiGenerateContent)]
    public void CitationSpecificInputOptionsAreRejectedWhenSurfaceCannotRepresentThem(
        LLMProvider provider,
        LLMApiSurface surface)
    {
        var request = Request("future-model", new DocumentContent
        {
            MediaType = "application/pdf",
            Source = new MediaSource { FileId = "file_123" },
            CitationsEnabled = true
        });

        var diagnostics = LLMRequestValidator.Validate(request, provider, surface);

        Assert.Contains(diagnostics, item =>
            item.Code == "request.document_citation_options.unrepresentable" &&
            item.Severity == RequestDiagnosticSeverity.Error);
    }

    [Fact]
    public void OpenAIFileCitationCreatesStableFileEvidenceSource()
    {
        const string json = """
        {"id":"r","model":"gpt","output":[{"type":"message","content":[{"type":"output_text","text":"See the file.","annotations":[{"type":"file_citation","file_id":"file_123","filename":"contract.pdf","index":0}]}]}]}
        """;

        var response = new OpenAIConverter().ConvertResponse(
            JsonSerializer.Deserialize<OpenAIResponse>(json)!);
        var message = Assert.Single(response.Choices).Message;
        var citation = Assert.Single(Assert.IsType<TextContent>(Assert.Single(message.Content)).Citations);
        var source = Assert.Single(message.Evidence.Sources);

        Assert.Equal("file_123", citation.FileId);
        Assert.Equal("contract.pdf", citation.FileName);
        Assert.Equal(source.Id, Assert.Single(citation.Sources).SourceId);
        Assert.Equal(EvidenceSourceKind.File, source.Kind);
        Assert.Equal("file_citation", citation.ProviderMetadata!["openai.annotationType"]);
    }

    [Fact]
    public void AnthropicCitationsAndStructuredOutputConflictIsExplicit()
    {
        var request = Request("claude-sonnet-4-6", new DocumentContent
        {
            MediaType = "application/pdf",
            Source = new MediaSource { FileId = "file_123" },
            CitationsEnabled = true
        });
        request.Output = new OutputFormat
        {
            Kind = OutputFormatKind.JsonSchema,
            JsonSchema = new JsonSchemaDefinition
            {
                Name = "answer",
                Schema = new Dictionary<string, object>
                {
                    ["type"] = "object",
                    ["properties"] = new Dictionary<string, object>
                    {
                        ["answer"] = new Dictionary<string, object> { ["type"] = "string" }
                    },
                    ["required"] = new[] { "answer" },
                    ["additionalProperties"] = false
                }
            }
        };

        var diagnostics = LLMRequestValidator.Validate(
            request, LLMProvider.Claude, LLMApiSurface.AnthropicMessages);

        Assert.Contains(diagnostics, item => item.Code == "claude.citations.structured_output_conflict");
    }

    private static UnifiedRequest Request(string model, params ContentBlock[] blocks) => new()
    {
        Model = model,
        Messages = new List<UnifiedMessage>
        {
            new(MessageRole.User, blocks.Length == 0
                ? new List<ContentBlock> { new TextContent { Text = "hello" } }
                : blocks.ToList())
        }
    };

    private static JsonNode Serialize<T>(T value) => JsonNode.Parse(JsonSerializer.Serialize(value,
        new JsonSerializerOptions { DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull }))!;
}
