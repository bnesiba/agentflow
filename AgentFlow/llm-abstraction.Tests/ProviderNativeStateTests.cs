using System.Text.Json;
using LLMAbstraction.Core.Models;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class ProviderNativeStateTests
{
    [Fact]
    public void NativeRepresentationOwnsACloneOfTheSourceJson()
    {
        ProviderNativeRepresentation representation;

        using (var document = JsonDocument.Parse("""{"type":"future","nested":{"value":42}}"""))
        {
            representation = ProviderNativeRepresentation.Create(
                ProviderIds.Claude,
                document.RootElement);
        }

        Assert.Equal(42, representation.Value
            .GetProperty("nested")
            .GetProperty("value")
            .GetInt32());
    }
}
