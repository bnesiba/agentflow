using System.Collections.Generic;
using System.Linq;

namespace LLMAbstraction.Core.Models
{
    /// <summary>
    /// Extension methods for unified message conversations.
    /// </summary>
    public static class UnifiedMessageExtensions
    {
        /// <summary>
        /// Checks whether the message is an assistant message with one or more tool calls.
        /// </summary>
        public static bool IsAssistantWithToolCalls(this UnifiedMessage? message)
        {
            return message?.Role == MessageRole.Assistant
                && message.Content.Any(content => content is ToolCallContent);
        }

        /// <summary>
        /// Checks whether the last message is an assistant message with one or more tool calls.
        /// </summary>
        public static bool LastMessageIsAssistantWithToolCalls(this IEnumerable<UnifiedMessage>? messages)
        {
            return messages?.LastOrDefault().IsAssistantWithToolCalls() == true;
        }
    }
}
