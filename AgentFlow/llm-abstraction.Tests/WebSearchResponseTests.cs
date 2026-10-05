using System.Text.Json;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Claude.Models;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.Gemini.Models;
using LLMAbstraction.Providers.OpenAI;
using LLMAbstraction.Providers.OpenAI.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class WebSearchResponseTests
{
    [Fact]
    public void OpenAIProjectsManagedSearchSourcesAndTextCitations()
    {
        const string json = """
        {
          "id":"resp_search","model":"gpt-5.6","status":"completed",
          "output":[
            {"id":"ws_1","type":"web_search_call","status":"completed","action":{"type":"search","query":"latest API","sources":[{"type":"url","url":"https://example.com/api","title":"API docs"}]},"results":[{"type":"image_result","image_url":"https://example.com/image.jpg","source_website_url":"https://example.com/page","thumbnail_url":"https://example.com/thumb.jpg","caption":"An API diagram"}]},
            {"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"The API changed.","annotations":[{"type":"url_citation","url":"https://example.com/api","title":"API docs","start_index":0,"end_index":15}]}]}
          ],
          "usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
        }
        """;

        var response = new OpenAIConverter().ConvertResponse(
            JsonSerializer.Deserialize<OpenAIResponse>(json)!);
        var message = Assert.Single(response.Choices).Message;
        var blocks = message.Content;

        var call = Assert.IsType<ProviderToolCallContent>(blocks[0]);
        Assert.Equal(ProviderToolCapability.WebSearch, call.Capability);
        Assert.Equal("ws_1", call.Id);
        var result = Assert.IsType<ProviderToolResultContent>(blocks[1]);
        Assert.Equal(2, result.Sources.Count);
        Assert.Equal("https://example.com/api", result.Sources[0].Url);
        Assert.Equal("https://example.com/image.jpg", result.Sources[1].ImageUrl);
        Assert.Equal("https://example.com/page", result.Sources[1].Url);
        var citation = Assert.Single(Assert.IsType<TextContent>(blocks[2]).Citations);
        Assert.Equal(15, citation.EndIndex);
        Assert.Equal("https://example.com/api", citation.Url);
        Assert.Equal(result.Sources[0].SourceId, Assert.Single(citation.Sources).SourceId);
        Assert.Equal(2, citation.AnswerSpan!.ContentBlockIndex);
        Assert.Equal(2, message.Evidence.Sources.Count);
        Assert.DoesNotContain(blocks, block => block is ToolCallContent);
    }

    [Fact]
    public void AnthropicProjectsServerSearchResultAndCitationsWithoutRequestingClientExecution()
    {
        const string json = """
        {
          "id":"msg_search","type":"message","role":"assistant","model":"claude-sonnet-4-6","stop_reason":"end_turn",
          "content":[
            {"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{"query":"latest API"}},
            {"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://example.com/api","title":"API docs","encrypted_content":"opaque","page_age":"August 4, 2026"}]},
            {"type":"text","text":"The API changed.","citations":[{"type":"web_search_result_location","url":"https://example.com/api","title":"API docs","cited_text":"API release notes"}]}
          ],
          "usage":{"input_tokens":1,"output_tokens":2,"server_tool_use":{"web_search_requests":1}}
        }
        """;

        var response = new ClaudeConverter().ConvertResponse(
            JsonSerializer.Deserialize<ClaudeMessageResponse>(json)!);
        var message = Assert.Single(response.Choices).Message;
        var blocks = message.Content;

        Assert.Equal("srvtoolu_1", Assert.IsType<ProviderToolCallContent>(blocks[0]).Id);
        var result = Assert.IsType<ProviderToolResultContent>(blocks[1]);
        Assert.Equal("srvtoolu_1", result.ToolCallId);
        var source = Assert.Single(result.Sources);
        Assert.Equal("https://example.com/api", source.Url);
        Assert.Equal("August 4, 2026", source.PageAge);
        Assert.Null(source.Snippet);
        var citation = Assert.Single(Assert.IsType<TextContent>(blocks[2]).Citations);
        Assert.Equal("https://example.com/api", citation.Url);
        Assert.Equal("API release notes", citation.CitedText);
        Assert.Equal(source.SourceId, Assert.Single(citation.Sources).SourceId);
        Assert.Single(message.Evidence.Sources);
        var serverUsage = Assert.IsType<Dictionary<string, int>>(
            response.Usage.ProviderMetadata!["claude.serverToolUse"]);
        Assert.Equal(1, serverUsage["web_search_requests"]);
        Assert.DoesNotContain(blocks, block => block is ToolCallContent);
    }

    [Fact]
    public void GeminiInteractionsProjectsSearchStepsSuggestionsAndInlineCitations()
    {
        const string json = """
        {
          "id":"int_search","model":"gemini-3.6-flash","status":"completed",
          "steps":[
            {"type":"google_search_call","id":"gs_1","arguments":{"queries":["latest API"]},"signature":"sig"},
            {"type":"google_search_result","call_id":"gs_1","result":{"search_suggestions":"<div>Search</div>"},"signature":"sig"},
            {"type":"model_output","content":[{"type":"text","text":"The API changed.","annotations":[{"type":"url_citation","url":"https://example.com/api","title":"API docs","start_index":0,"end_index":15}]}]}
          ]
        }
        """;

        var response = new GeminiInteractionsConverter().ConvertResponse(
            JsonSerializer.Deserialize<GeminiInteractionResponse>(json)!);
        var message = Assert.Single(response.Choices).Message;
        var blocks = message.Content;

        Assert.Equal("gs_1", Assert.IsType<ProviderToolCallContent>(blocks[0]).Id);
        var result = Assert.IsType<ProviderToolResultContent>(blocks[1]);
        Assert.Equal("<div>Search</div>", result.SearchSuggestionsHtml);
        var citation = Assert.Single(Assert.IsType<TextContent>(blocks[2]).Citations);
        Assert.Equal("https://example.com/api", citation.Url);
        var artifact = Assert.Single(message.Evidence.AttributionArtifacts);
        Assert.Equal(AttributionArtifactKind.Html, artifact.Kind);
        Assert.True(artifact.DisplayRequired);
        Assert.Equal("<div>Search</div>", artifact.Content);
        Assert.DoesNotContain(blocks, block => block is ToolCallContent);
    }

    [Fact]
    public void LegacyGeminiProjectsGroundingSourcesSupportsAndRequiredSuggestionsMarkup()
    {
        const string json = """
        {
          "responseId":"gem_search","modelVersion":"gemini-3.6-flash",
          "candidates":[{
            "index":0,"finishReason":"STOP","content":{"role":"model","parts":[{"text":"The API changed."}]},
            "groundingMetadata":{
              "webSearchQueries":["latest API"],
              "searchEntryPoint":{"renderedContent":"<div>Search</div>"},
              "groundingChunks":[{"web":{"uri":"https://example.com/api","title":"API docs"}},{"web":{"uri":"https://example.com/release","title":"Release notes"}}],
              "groundingSupports":[{"segment":{"startIndex":0,"endIndex":15,"text":"The API changed."},"groundingChunkIndices":[0,1],"confidenceScores":[0.95,0.75]}]
            }
          }]
        }
        """;

        var response = new GeminiConverter().ConvertResponse(
            JsonSerializer.Deserialize<GeminiGenerateResponse>(json)!);
        var blocks = Assert.Single(response.Choices).Message.Content;

        var text = Assert.IsType<TextContent>(blocks[0]);
        Assert.Equal(2, text.Citations.Count);
        Assert.Equal("https://example.com/api", text.Citations[0].Url);
        Assert.IsType<ProviderToolCallContent>(blocks[1]);
        var result = Assert.IsType<ProviderToolResultContent>(blocks[2]);
        Assert.Equal(2, result.Sources.Count);
        Assert.Equal("https://example.com/api", result.Sources[0].Url);
        Assert.Equal("<div>Search</div>", result.SearchSuggestionsHtml);
        var evidence = Assert.Single(response.Choices).Message.Evidence;
        var grounding = Assert.Single(evidence.GroundingSupports);
        Assert.Equal(2, grounding.Sources.Count);
        Assert.Equal(new[] { 0.95, 0.75 }, grounding.SourceConfidences);
        Assert.Equal(0, grounding.AnswerSpan!.StartIndex);
        Assert.Equal(15, grounding.AnswerSpan.EndIndex);
        Assert.Equal(2, evidence.Sources.Count);
        Assert.True(Assert.Single(evidence.AttributionArtifacts).DisplayRequired);
    }

    [Fact]
    public void AnthropicCitationLocationsRemainTypedAndUnambiguous()
    {
        const string json = """
        {
          "id":"msg_citations","type":"message","role":"assistant","model":"claude-sonnet-4-6","stop_reason":"end_turn",
          "content":[{"type":"text","text":"Cited answer","citations":[
            {"type":"char_location","document_index":0,"document_title":"Manual","file_id":"file_1","start_char_index":10,"end_char_index":20,"cited_text":"characters"},
            {"type":"page_location","document_index":0,"document_title":"Manual","file_id":"file_1","start_page_number":2,"end_page_number":4,"cited_text":"pages"},
            {"type":"content_block_location","document_index":0,"document_title":"Manual","file_id":"file_1","start_block_index":1,"end_block_index":3,"cited_text":"blocks"},
            {"type":"search_result_location","search_result_index":2,"source":"kb","title":"KB result","start_block_index":0,"end_block_index":1,"cited_text":"result"}
          ]}],
          "usage":{"input_tokens":1,"output_tokens":2}
        }
        """;

        var message = Assert.Single(new ClaudeConverter().ConvertResponse(
            JsonSerializer.Deserialize<ClaudeMessageResponse>(json)!).Choices).Message;
        var citations = Assert.IsType<TextContent>(Assert.Single(message.Content)).Citations;

        var characters = Assert.IsType<CharacterRangeLocation>(citations[0].SourceLocation);
        Assert.Equal(10, characters.StartIndex);
        Assert.Equal(20, characters.EndIndex);
        Assert.True(characters.EndExclusive);
        var pages = Assert.IsType<PageRangeLocation>(citations[1].SourceLocation);
        Assert.Equal(2, pages.StartPage);
        Assert.Equal(4, pages.EndPage);
        Assert.Null(pages.EndInclusive);
        var blocks = Assert.IsType<ContentBlockRangeLocation>(citations[2].SourceLocation);
        Assert.Equal(1, blocks.StartBlockIndex);
        Assert.Equal(3, blocks.EndBlockIndex);
        Assert.True(blocks.EndExclusive);
        Assert.Equal(0, citations[0].ProviderMetadata!["claude.documentIndex"]);
        Assert.Equal(2, citations[3].ProviderMetadata!["claude.searchResultIndex"]);
        Assert.Equal("kb", citations[3].ProviderMetadata!["claude.source"]);
        Assert.All(citations, citation => Assert.Single(citation.Sources));
        Assert.Equal(2, message.Evidence.Sources.Count);
    }

    [Fact]
    public void AnthropicPauseTurnIsDistinctFromFunctionToolExecution()
    {
        var native = new ClaudeMessageResponse
        {
            Id = "msg_pause",
            Model = "claude-sonnet-4-6",
            StopReason = "pause_turn",
            Content =
            {
                new ClaudeContentBlock
                {
                    Type = "server_tool_use",
                    Id = "srvtoolu_1",
                    Name = "web_search"
                }
            }
        };

        var choice = Assert.Single(new ClaudeConverter().ConvertResponse(native).Choices);
        Assert.Equal(FinishReason.Pause, choice.FinishReason);
        Assert.IsType<ProviderToolCallContent>(Assert.Single(choice.Message.Content));
    }
}
