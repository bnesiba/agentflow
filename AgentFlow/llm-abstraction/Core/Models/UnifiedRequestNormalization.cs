using System;
using System.Collections.Generic;
using System.Linq;

namespace LLMAbstraction.Core.Models
{
    internal static class UnifiedRequestNormalization
    {
        public static string? CombineInstructions(UnifiedRequest request)
        {
            var sections = new List<string>();
            if (!string.IsNullOrWhiteSpace(request.Instructions))
                sections.Add(request.Instructions);

            foreach (var message in request.Messages.Where(message => message.Role == MessageRole.System))
            {
                var textParts = new List<string>();
                foreach (var block in message.Content)
                {
                    if (block is not TextContent text)
                    {
                        throw new NotSupportedException(
                            $"System message content type '{block.Type}' cannot be represented by the unified instruction channel.");
                    }

                    textParts.Add(text.Text);
                }

                if (textParts.Count > 0)
                    sections.Add(string.Join("\n", textParts));
            }

            return sections.Count == 0 ? null : string.Join("\n\n", sections);
        }
    }
}
