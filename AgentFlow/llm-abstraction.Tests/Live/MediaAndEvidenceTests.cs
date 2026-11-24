using LLMAbstraction.Core.Models;
using Xunit;
namespace LLMAbstraction.Tests.Live;

public partial class LiveProviderTests
{
    private static string Fixture(string name) => Convert.ToBase64String(
        File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Live", "Fixtures", name)));

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    public async Task Remote_image_is_read(Surface surface)
    {
        var request = Request(surface, "Transcribe the word in this image.");
        request.Messages[0].Content.Add(new ImageContent
        {
            Source = new() { MediaType = "image/png",
                Url = "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png" }
        });
        var message = Assert.Single((await Service(surface).GenerateAsync(request)).Choices).Message;
        Assert.Contains("Google", Text(message), StringComparison.OrdinalIgnoreCase);
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Inline_image_is_read(Surface surface)
    {
        var request = Request(surface, "Name the dominant color in one word.");
        request.Messages[0].Content.Add(new ImageContent
        {
            Source = new() { MediaType = "image/png", Data = Fixture("swatch.png") }
        });
        var message = Assert.Single((await Service(surface).GenerateAsync(request)).Choices).Message;
        Assert.Contains("red", Text(message), StringComparison.OrdinalIgnoreCase);
    }

    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Pdf_is_read_and_cited_where_supported(Surface surface)
    {
        var request = Request(surface, surface == Surface.Claude
            ? "Give the ticket code and cite the document." : "Give the ticket code.");
        request.Messages[0].Content.Insert(0, new DocumentContent
        {
            MediaType = "application/pdf",
            Source = new() { FileName = "ticket.pdf", Base64Data = Fixture("ticket.pdf") },
            CitationsEnabled = surface == Surface.Claude ? true : null
        });
        var message = Assert.Single((await Service(surface).GenerateAsync(request)).Choices).Message;
        Assert.Contains("ORCHID-817", Text(message));
        if (surface == Surface.Claude)
            Assert.NotEmpty(message.Content.OfType<TextContent>().SelectMany(t => t.Citations));
    }

    [LiveTheory]
    [InlineData(Surface.GeminiInteractions, "beeps.wav", "audio/wav", "Count the beeps. Answer with one digit.", "3")]
    [InlineData(Surface.GeminiGenerateContent, "beeps.wav", "audio/wav", "Count the beeps. Answer with one digit.", "3")]
    [InlineData(Surface.GeminiInteractions, "swatch.avi", "video/avi", "Name the dominant color.", "blue")]
    [InlineData(Surface.GeminiGenerateContent, "swatch.avi", "video/avi", "Name the dominant color.", "blue")]
    public async Task Inline_audio_or_video_is_read(Surface surface, string file, string mime, string question, string expected)
    {
        var request = Request(surface, question);
        request.Messages[0].Content.Add(new MediaContent
        {
            MediaType = mime, Source = new() { Base64Data = Fixture(file) }
        });
        var message = Assert.Single((await Service(surface).GenerateAsync(request)).Choices).Message;
        Assert.Contains(expected, Text(message), StringComparison.OrdinalIgnoreCase);
    }

    [LiveTheory]
    [InlineData(Surface.Claude)]
    public async Task Supplied_search_result_is_cited(Surface surface)
    {
        var request = Request(surface, "Which gate? Cite the supplied search result.");
        request.Messages[0].Content.Insert(0, new SearchResultContent
        {
            Title = "Schedule", Source = "https://example.org/schedule", CitationsEnabled = true,
            Content = [new() { Text = "The departure gate is B79." }]
        });
        var message = Assert.Single((await Service(surface).GenerateAsync(request)).Choices).Message;
        Assert.Contains("B79", Text(message));
        Assert.NotEmpty(message.Content.OfType<TextContent>().SelectMany(t => t.Citations));
        Assert.NotEmpty(message.Evidence.Sources);
    }
}
