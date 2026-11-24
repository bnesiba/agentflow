using LLMAbstraction.Core.Models;
using Xunit;
using LLMAbstraction.Core.Interfaces;

namespace LLMAbstraction.Tests.Live;

public partial class LiveProviderTests
{
    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    public async Task Repeated_prefix_reports_cached_tokens(Surface surface)
    {
        var request = Request(surface, "Reply OK.");
        var prefix = new TextContent
        {
            Text = string.Concat(Enumerable.Repeat(" stable", surface == Surface.OpenAI ? 1100 : 560)),
            Cache = new()
        };
        request.Messages[0].Content.Insert(0, prefix);
        request.Cache = new()
        {
            Mode = PromptCacheMode.ExplicitBreakpointsOnly,
            OpenAI = surface == Surface.OpenAI ? new() { CacheKey = "agentflow-live" } : null
        };
        var service = Service(surface);
        var first = await service.GenerateAsync(request);
        Assert.True(first.Usage.CacheWriteTokens > 0 || first.Usage.CacheReadTokens > 0);
        var second = await service.GenerateAsync(request);
        Assert.True(second.Usage.CacheReadTokens > 0);
    }

    [LiveTheory]
    [InlineData(Surface.GeminiInteractions)]
    public async Task Gemini_cache_create_get_list_update_use_and_delete(Surface surface)
    {
        var service = Service(surface);
        var caches = Assert.IsAssignableFrom<IPromptCacheService>(service);
        var prefix = Request(surface, "The ticket code is FERN-712." + string.Concat(Enumerable.Repeat(" stable", 4200)));
        prefix.Model = "gemini-3.5-flash-lite";
        var created = await caches.CreatePromptCacheAsync(new()
        {
            Prefix = prefix, DisplayName = "agentflow-live", Ttl = TimeSpan.FromMinutes(3)
        });
        try
        {
            var cached = await caches.GetPromptCacheAsync(created.Name);
            Assert.Equal(created.Name, cached.Name);
            Assert.True(cached.CachedTokens > 0);
            string? cursor = null;
            PromptCacheResourcePage page;
            do
            {
                page = await caches.ListPromptCachesAsync(pageSize: 100, pageToken: cursor);
                cursor = page.NextPageToken;
            } while (!page.Items.Any(item => item.Name == created.Name) && !string.IsNullOrEmpty(cursor));
            Assert.Contains(page.Items, item => item.Name == created.Name);
            var updated = await caches.UpdatePromptCacheExpirationAsync(created.Name,
                new() { Ttl = TimeSpan.FromMinutes(5) });
            Assert.True(updated.ExpireTime > cached.ExpireTime);

            var followUp = Request(surface, "Copy the complete ticket code exactly as written, including the hyphen and every digit. Reply with only the code.");
            followUp.Model = prefix.Model;
            followUp.Parameters.MaxOutputTokens = 512;
            followUp.Cache = new() { Gemini = new() { CachedContentName = created.Name } };
            var response = await service.GenerateAsync(followUp);
            Assert.True(response.Usage.CacheReadTokens > 0);
            Assert.Contains("FERN-712", Text(Assert.Single(response.Choices).Message));
        }
        finally
        {
            await caches.DeletePromptCacheAsync(created.Name);
        }
    }
}
