using System;
using System.Collections.Generic;
using System.Linq;

namespace LLMAbstraction.Core.Models
{
    public sealed class EvidenceCollection
    {
        public List<EvidenceSource> Sources { get; set; } = new();
        public List<GroundingSupport> GroundingSupports { get; set; } = new();
        public List<AttributionArtifact> AttributionArtifacts { get; set; } = new();
    }

    public enum EvidenceSourceKind
    {
        Web,
        Image,
        File,
        Document,
        SearchResult,
        Place,
        Media,
        Unknown
    }

    public sealed class EvidenceSource
    {
        public string Id { get; set; } = string.Empty;
        public EvidenceSourceKind Kind { get; set; }
        public string? Title { get; set; }
        public string? Uri { get; set; }
        public string? FileId { get; set; }
        public string? FileName { get; set; }
        public string? Excerpt { get; set; }
        public List<SourceLocation> Locations { get; set; } = new();
        public Dictionary<string, object>? ProviderMetadata { get; set; }
        public ProviderNativeRepresentation? NativeRepresentation { get; set; }
    }

    /// <summary>
    /// Typed location within evidence. Each range states its own end-boundary
    /// semantics rather than assuming providers index every medium alike.
    /// </summary>
    public abstract class SourceLocation
    {
        public abstract string Type { get; }
    }

    public enum TextIndexUnit
    {
        Utf16CodeUnits,
        UnicodeCodePoints,
        Utf8Bytes,
        ProviderDefined
    }

    public sealed class CharacterRangeLocation : SourceLocation
    {
        public override string Type => "character_range";
        public int StartIndex { get; set; }
        public int EndIndex { get; set; }
        public TextIndexUnit Unit { get; set; } = TextIndexUnit.ProviderDefined;
        public bool EndExclusive { get; set; } = true;
    }

    public sealed class PageRangeLocation : SourceLocation
    {
        public override string Type => "page_range";
        public int StartPage { get; set; }
        public int EndPage { get; set; }
        public bool? EndInclusive { get; set; }
    }

    public sealed class ContentBlockRangeLocation : SourceLocation
    {
        public override string Type => "content_block_range";
        public int StartBlockIndex { get; set; }
        public int EndBlockIndex { get; set; }
        public bool EndExclusive { get; set; } = true;
    }

    public sealed class TimestampRangeLocation : SourceLocation
    {
        public override string Type => "timestamp_range";
        public TimeSpan Start { get; set; }
        public TimeSpan End { get; set; }
        public bool? EndInclusive { get; set; }
    }

    public sealed class UriFragmentLocation : SourceLocation
    {
        public override string Type => "uri_fragment";
        public string Fragment { get; set; } = string.Empty;
    }

    /// <summary>
    /// Range in one returned text content block. EndIndex is exclusive.
    /// </summary>
    public sealed class AnswerTextSpan
    {
        public int ContentBlockIndex { get; set; }
        public int StartIndex { get; set; }
        public int EndIndex { get; set; }
        public TextIndexUnit Unit { get; set; } = TextIndexUnit.ProviderDefined;
    }

    public sealed class CitationSourceReference
    {
        public string SourceId { get; set; } = string.Empty;
        public SourceLocation? Location { get; set; }
    }

    public sealed class GroundingSupport
    {
        public AnswerTextSpan? AnswerSpan { get; set; }
        public List<CitationSourceReference> Sources { get; set; } = new();
        public double? Confidence { get; set; }
        public IReadOnlyList<double>? SourceConfidences { get; set; }
        public Dictionary<string, object>? ProviderMetadata { get; set; }
        public ProviderNativeRepresentation? NativeRepresentation { get; set; }
    }

    public enum AttributionArtifactKind
    {
        Html,
        Text,
        Image,
        Widget,
        Unknown
    }

    public sealed class AttributionArtifact
    {
        public string Id { get; set; } = string.Empty;
        public AttributionArtifactKind Kind { get; set; }
        public string Content { get; set; } = string.Empty;
        public string? MimeType { get; set; }
        public bool DisplayRequired { get; set; }
        public ProviderNativeRepresentation? NativeRepresentation { get; set; }
    }

    internal static class EvidenceProjector
    {
        public static void Project(UnifiedMessage message, string provider)
        {
            // Sources must be stable before citation references are linked.
            foreach (var result in message.Content.OfType<ProviderToolResultContent>())
                ProjectResult(message, result, provider);

            for (var blockIndex = 0; blockIndex < message.Content.Count; blockIndex++)
            {
                if (message.Content[blockIndex] is TextContent text)
                    foreach (var citation in text.Citations)
                        ProjectCitation(message, citation, provider, blockIndex);
            }
        }

        private static void ProjectResult(
            UnifiedMessage message,
            ProviderToolResultContent result,
            string provider)
        {
            for (var index = 0; index < result.Sources.Count; index++)
            {
                var web = result.Sources[index];
                web.SourceId ??= $"{provider}:{result.ToolCallId}:source:{index}";
                if (message.Evidence.Sources.Any(source => source.Id == web.SourceId))
                    continue;
                message.Evidence.Sources.Add(new EvidenceSource
                {
                    Id = web.SourceId,
                    Kind = web.SourceType?.Contains("image", StringComparison.OrdinalIgnoreCase) == true
                        ? EvidenceSourceKind.Image
                        : EvidenceSourceKind.Web,
                    Uri = web.Url,
                    Title = web.Title,
                    Excerpt = web.Snippet,
                    ProviderMetadata = BuildWebMetadata(web)
                });
            }

            if (!string.IsNullOrEmpty(result.SearchSuggestionsHtml))
            {
                var id = $"{provider}:{result.ToolCallId}:search_suggestions";
                if (!message.Evidence.AttributionArtifacts.Any(artifact => artifact.Id == id))
                {
                    message.Evidence.AttributionArtifacts.Add(new AttributionArtifact
                    {
                        Id = id,
                        Kind = AttributionArtifactKind.Html,
                        Content = result.SearchSuggestionsHtml,
                        MimeType = "text/html",
                        DisplayRequired = true,
                        NativeRepresentation = result.NativeRepresentation
                    });
                }
            }
        }

        private static void ProjectCitation(
            UnifiedMessage message,
            Citation citation,
            string provider,
            int blockIndex)
        {
            if (citation.AnswerSpan == null && citation.StartIndex is int start && citation.EndIndex is int end)
            {
                citation.AnswerSpan = new AnswerTextSpan
                {
                    ContentBlockIndex = blockIndex,
                    StartIndex = start,
                    EndIndex = end,
                    Unit = TextIndexUnit.ProviderDefined
                };
            }
            if (citation.Sources.Count > 0)
                return;

            var source = message.Evidence.Sources.FirstOrDefault(existing =>
                (!string.IsNullOrEmpty(citation.Url) && existing.Uri == citation.Url) ||
                (!string.IsNullOrEmpty(citation.FileId) && existing.FileId == citation.FileId) ||
                (!string.IsNullOrEmpty(citation.FileName) && existing.FileName == citation.FileName) ||
                (citation.Url == null && citation.FileId == null && citation.FileName == null &&
                    !string.IsNullOrEmpty(citation.Title) && existing.Title == citation.Title));
            if (source == null)
            {
                source = new EvidenceSource
                {
                    Id = $"{provider}:citation:{message.Evidence.Sources.Count}",
                    Kind = citation.Url != null
                        ? EvidenceSourceKind.Web
                        : citation.FileId != null || citation.FileName != null
                            ? EvidenceSourceKind.File
                            : EvidenceSourceKind.Document,
                    Uri = citation.Url,
                    Title = citation.Title,
                    FileId = citation.FileId,
                    FileName = citation.FileName,
                    Excerpt = citation.CitedText,
                    ProviderMetadata = citation.ProviderMetadata,
                    NativeRepresentation = citation.NativeRepresentation
                };
                message.Evidence.Sources.Add(source);
            }

            SourceLocation? location = citation.SourceLocation ?? (citation.PageNumber is int page
                ? new PageRangeLocation { StartPage = page, EndPage = page }
                : null);
            citation.Sources.Add(new CitationSourceReference
            {
                SourceId = source.Id,
                Location = location
            });
        }

        private static Dictionary<string, object>? BuildWebMetadata(WebSource source)
        {
            var metadata = new Dictionary<string, object>();
            if (source.ImageUrl != null) metadata["imageUrl"] = source.ImageUrl;
            if (source.ThumbnailUrl != null) metadata["thumbnailUrl"] = source.ThumbnailUrl;
            if (source.Caption != null) metadata["caption"] = source.Caption;
            if (source.PageAge != null) metadata["pageAge"] = source.PageAge;
            return metadata.Count == 0 ? null : metadata;
        }
    }
}
