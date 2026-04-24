using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// Represents the role of a message in a conversation
    /// </summary>
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum MessageRole
    {
        System,
        User,
        Assistant,
        Tool
    }

    /// <summary>
    /// Unified message structure that works across all LLM providers
    /// </summary>
    public class UnifiedMessage
    {
        public MessageRole Role { get; set; }
        public List<ContentBlock> Content { get; set; } = new();

        // Convenience constructor for simple text messages
        public UnifiedMessage(MessageRole role, string text)
        {
            Role = role;
            Content = new List<ContentBlock>
            {
                new TextContent { Text = text }
            };
        }

        public UnifiedMessage(MessageRole role, List<ContentBlock> content)
        {
            Role = role;
            Content = content;
        }

        public UnifiedMessage() { }
    }

    /// <summary>
    /// Base class for all content block types
    /// </summary>
    public abstract class ContentBlock
    {
        public abstract string Type { get; }
    }

    /// <summary>
    /// Text content block
    /// </summary>
    public class TextContent : ContentBlock
    {
        public override string Type => "text";
        public string Text { get; set; } = string.Empty;
    }

    /// <summary>
    /// Image content block
    /// </summary>
    public class ImageContent : ContentBlock
    {
        public override string Type => "image";
        public ImageSource Source { get; set; } = new();
    }

    /// <summary>
    /// Image source (base64 or URL)
    /// </summary>
    public class ImageSource
    {
        public string? MediaType { get; set; }
        public string? Data { get; set; }  // Base64 encoded
        public string? Url { get; set; }
    }

    /// <summary>
    /// Tool call content block (when model wants to call a tool)
    /// </summary>
    public class ToolCallContent : ContentBlock
    {
        public override string Type => "tool_call";
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public Dictionary<string, object> Input { get; set; } = new();
    }

    /// <summary>
    /// Tool result content block (response from tool execution)
    /// </summary>
    public class ToolResultContent : ContentBlock
    {
        public override string Type => "tool_result";
        public string ToolCallId { get; set; } = string.Empty;
        public string Output { get; set; } = string.Empty;
        public bool? IsError { get; set; }
    }
}
