using LLMAbstraction.Core.Models;
using Xunit;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;

namespace LLMAbstraction.Tests.Live;

public partial class LiveProviderTests
{
    [LiveTheory]
    [InlineData(Surface.OpenAI)]
    [InlineData(Surface.Claude)]
    [InlineData(Surface.GeminiInteractions)]
    [InlineData(Surface.GeminiGenerateContent)]
    public async Task Uploaded_document_reference_is_read(Surface surface)
    {
        using var files = new HttpClient();
        var bytes = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Live", "Fixtures", "ticket.pdf"));
        MediaSource source;
        string deletePath;
        string? fileState = null;
        if (surface is Surface.OpenAI or Surface.Claude)
        {
            bool openAI = surface == Surface.OpenAI;
            files.BaseAddress = new Uri(openAI ? "https://api.openai.com/v1/" : "https://api.anthropic.com/v1/");
            using var form = new MultipartFormDataContent();
            if (openAI)
            {
                files.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", Key(surface));
                form.Add(new StringContent("user_data"), "purpose");
            }
            else
            {
                files.DefaultRequestHeaders.Add("x-api-key", Key(surface));
                files.DefaultRequestHeaders.Add("anthropic-version", "2023-06-01");
            }
            var content = new ByteArrayContent(bytes);
            content.Headers.ContentType = new MediaTypeHeaderValue("application/pdf");
            form.Add(content, "file", "ticket.pdf");
            using var upload = await files.PostAsync("files", form);
            upload.EnsureSuccessStatusCode();
            var file = await upload.Content.ReadFromJsonAsync<JsonElement>();
            source = new() { FileId = file.GetProperty("id").GetString()! };
            deletePath = "files/" + Uri.EscapeDataString(source.FileId);
        }
        else
        {
            files.BaseAddress = new Uri("https://generativelanguage.googleapis.com/");
            using var start = new HttpRequestMessage(HttpMethod.Post, "upload/v1beta/files")
            {
                Content = JsonContent.Create(new { file = new { display_name = "agentflow-live-ticket" } })
            };
            start.Headers.Add("x-goog-api-key", Key(surface));
            start.Headers.Add("X-Goog-Upload-Protocol", "resumable");
            start.Headers.Add("X-Goog-Upload-Command", "start");
            start.Headers.Add("X-Goog-Upload-Header-Content-Length", bytes.Length.ToString());
            start.Headers.Add("X-Goog-Upload-Header-Content-Type", "application/pdf");
            using var session = await files.SendAsync(start);
            session.EnsureSuccessStatusCode();
            using var finish = new HttpRequestMessage(HttpMethod.Post, session.Headers.GetValues("X-Goog-Upload-Url").Single())
            {
                Content = new ByteArrayContent(bytes)
            };
            finish.Headers.Add("X-Goog-Upload-Offset", "0");
            finish.Headers.Add("X-Goog-Upload-Command", "upload, finalize");
            using var upload = await files.SendAsync(finish);
            upload.EnsureSuccessStatusCode();
            var body = await upload.Content.ReadFromJsonAsync<JsonElement>();
            var file = body.GetProperty("file");
            source = new() { FileUri = file.GetProperty("uri").GetString()! };
            deletePath = "v1beta/" + file.GetProperty("name").GetString();
            fileState = file.GetProperty("state").GetString();
            files.DefaultRequestHeaders.Add("x-goog-api-key", Key(surface));
        }

        try
        {
            if (fileState != null)
            {
                using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(60));
                while (fileState == "PROCESSING")
                {
                    await Task.Delay(TimeSpan.FromSeconds(1), deadline.Token);
                    var file = await files.GetFromJsonAsync<JsonElement>(deletePath, deadline.Token);
                    fileState = file.GetProperty("state").GetString();
                }
                Assert.Equal("ACTIVE", fileState);
            }
            var request = Request(surface, "What is the ticket code?");
            request.Messages[0].Content.Add(new DocumentContent { MediaType = "application/pdf", Source = source });
            var response = await Service(surface).GenerateAsync(request);
            Assert.Contains("ORCHID-817", Text(Assert.Single(response.Choices).Message));
        }
        finally
        {
            using var deletion = await files.DeleteAsync(deletePath);
            deletion.EnsureSuccessStatusCode();
        }
    }
}
