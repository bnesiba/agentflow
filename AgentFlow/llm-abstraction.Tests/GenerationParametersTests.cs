using System.Text.Json;
using System.Text.Json.Serialization;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Providers.Claude;
using LLMAbstraction.Providers.Gemini;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public class GenerationParametersTests
{
    [Fact]
    public void DefaultRequestsOmitSamplingFieldsOnEverySurface()
    {
        var request = new UnifiedRequest
        {
            Model = "future-model",
            Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
        };
        object[] converted =
        {
            new OpenAIConverter().ConvertRequest(request),
            new ClaudeConverter().ConvertRequest(request),
            new GeminiConverter().ConvertRequest(request),
            new GeminiInteractionsConverter().ConvertRequest(request)
        };
        foreach (var native in converted)
        {
            var json = JsonSerializer.Serialize(native, native.GetType(), new JsonSerializerOptions
            {
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
            });
            foreach (var field in new[] { "temperature", "top_p", "top_k", "topP", "topK" })
                Assert.DoesNotContain($"\"{field}\":", json);
        }
    }
}
