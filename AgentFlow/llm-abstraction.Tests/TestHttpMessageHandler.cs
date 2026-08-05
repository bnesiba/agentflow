using System.Net;
using System.Text;

namespace LLMAbstraction.Tests;

internal sealed class TestHttpMessageHandler(
    string responseBody,
    HttpStatusCode statusCode = HttpStatusCode.OK,
    IReadOnlyDictionary<string, string>? headers = null) : HttpMessageHandler
{
    public HttpRequestMessage? LastRequest { get; private set; }
    public string? LastRequestBody { get; private set; }

    protected override async Task<HttpResponseMessage> SendAsync(
        HttpRequestMessage request,
        CancellationToken cancellationToken)
    {
        LastRequest = request;
        LastRequestBody = request.Content == null
            ? null
            : await request.Content.ReadAsStringAsync(cancellationToken);
        var response = new HttpResponseMessage(statusCode)
        {
            Content = new StringContent(responseBody, Encoding.UTF8, "text/event-stream")
        };
        if (headers != null)
        {
            foreach (var header in headers)
                response.Headers.TryAddWithoutValidation(header.Key, header.Value);
        }
        return response;
    }
}
