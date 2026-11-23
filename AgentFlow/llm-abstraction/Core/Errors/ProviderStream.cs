using System.Runtime.CompilerServices;
using System.Text.Json;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Transport;

namespace LLMAbstraction.Core.Errors;

internal static class ProviderStream
{
    public static async IAsyncEnumerable<StreamChunk> Observe(
        IAsyncEnumerable<StreamChunk> source,
        LLMProvider provider,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var partial = false;
        var tools = false;
        var terminal = false;
        var metadata = new TransportMetadata();
        await using var iterator = source.GetAsyncEnumerator(cancellationToken);
        while (true)
        {
            bool available;
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                available = await iterator.MoveNextAsync().ConfigureAwait(false);
                if (!available && !terminal)
                    throw new IOException("Provider stream ended before a terminal event.");
            }
            catch (LLMApiException exception)
            {
                exception.HasPartialOutput = partial;
                exception.HasToolCalls = tools;
                throw;
            }
            catch (LLMTransportException exception)
            {
                exception.HasPartialOutput = partial;
                exception.HasToolCalls = tools;
                throw;
            }
            catch (Exception exception) when (exception is IOException or HttpRequestException or
                JsonException or OperationCanceledException ||
                exception is InvalidOperationException { InnerException: JsonException })
            {
                cancellationToken.ThrowIfCancellationRequested();
                throw new LLMTransportException(provider,
                    exception is OperationCanceledException ? $"{provider} stream timed out." : $"{provider} stream could not be completed.",
                    exception is IOException or HttpRequestException or OperationCanceledException, metadata, exception)
                { HasPartialOutput = partial, HasToolCalls = tools };
            }
            if (!available) yield break;
            var chunk = iterator.Current;
            metadata = chunk.Transport ?? metadata;
            partial |= !string.IsNullOrEmpty(chunk.Delta.Content) || chunk.Delta.ContentBlocks?.Count > 0 || chunk.Delta.ToolCalls?.Count > 0;
            tools |= chunk.Delta.ToolCalls?.Count > 0 || chunk.Delta.ContentBlocks?.Any(block => block is ToolCallContent or ProviderToolCallContent) == true;
            terminal |= chunk.FinishReason != null;
            yield return chunk;
        }
    }
}
