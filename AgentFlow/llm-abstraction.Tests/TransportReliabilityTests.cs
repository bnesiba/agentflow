using System.Net;
using System.Text;
using LLMAbstraction.Core;
using LLMAbstraction.Core.Errors;
using LLMAbstraction.Core.Models;
using LLMAbstraction.Core.Transport;
using LLMAbstraction.Providers.OpenAI;
using Xunit;

namespace LLMAbstraction.Tests;

public sealed class TransportReliabilityTests
{
    [Fact]
    public async Task RateLimitErrorPreservesProviderMessage()
    {
        var handler = new SequenceHandler(_ => Response(HttpStatusCode.TooManyRequests,
            """{"error":{"message":"slow down"}}"""));
        using var client = Client(handler);
        var service = new OpenAIService(client, "key");
        var error = await Assert.ThrowsAsync<LLMApiException>(() => service.GenerateAsync(Request()));
        Assert.Equal(1, handler.CallCount);
    }

    [Fact]
    public async Task NonTransientStatusIsSentOnce()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.BadRequest,
                """{"error":{"message":"bad request","type":"invalid_request_error"}}"""));
        using var client = Client(handler);
        var service = new OpenAIService(client, "key");

        var exception = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(Request()));

        Assert.Equal(1, handler.CallCount);
        Assert.Single(exception.Transport!.Attempts);
    }

    [Fact]
    public async Task TransientResponseRecordsOneAttempt()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"));
        using var client = Client(handler);
        var service = new OpenAIService(client, "key");

        var exception = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(Request()));

        Assert.Equal(1, handler.CallCount);
        Assert.Single(exception.Transport!.Attempts);
    }

    [Fact]
    public async Task ConnectionFailureProducesStructuredFailure()
    {
        var handler = new SequenceHandler(
            _ => throw new HttpRequestException("connection reset"),
            _ => throw new HttpRequestException("connection reset"),
            _ => throw new HttpRequestException("connection reset"));
        using var client = Client(handler);
        var service = new OpenAIService(client, "key");

        var exception = await Assert.ThrowsAsync<LLMTransportException>(
            () => service.GenerateAsync(Request()));

        Assert.Equal(LLMProvider.OpenAI, exception.Provider);
        Assert.True(exception.IsTransient);
        Assert.Single(exception.Transport.Attempts);
        Assert.All(exception.Transport.Attempts,
            attempt => Assert.Contains(nameof(HttpRequestException), attempt.ExceptionType));
    }

    [Fact]
    public async Task StreamInitialFailureIsSentOnce()
    {
        var handler = new SequenceHandler(_ => Response(HttpStatusCode.BadGateway, "bad gateway"));
        using var client = Client(handler);
        var service = new OpenAIService(client, "key");
        await Assert.ThrowsAsync<LLMApiException>(async () =>
        {
            await foreach (var chunk in service.StreamAsync(Request())) { }
        });
        Assert.Equal(1, handler.CallCount);
    }

    [Fact]
    public async Task AdmissionLeaseIsReleasedAfterFailure()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.OK, SuccessfulResponse));
        var admission = new RecordingAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            AdmissionPolicy = admission
        });

        Assert.False(admission.LeaseActive);
        await Assert.ThrowsAsync<LLMApiException>(() => service.GenerateAsync(Request()));

        Assert.Equal(1, admission.Acquisitions);
        Assert.Equal(1, admission.Disposals);
        Assert.False(admission.LeaseActive);
        Assert.Equal(1, handler.CallCount);
    }

    [Fact]
    public async Task StreamingAdmissionLeaseIsHeldUntilEnumeratorIsDisposed()
    {
        var completed = """
        data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.6","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":0,"total_tokens":1}}}

        """;
        var handler = new SequenceHandler(_ => Response(HttpStatusCode.OK, completed));
        var admission = new RecordingAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            AdmissionPolicy = admission
        });

        await using var enumerator = service.StreamAsync(Request()).GetAsyncEnumerator();
        Assert.True(await enumerator.MoveNextAsync());
        Assert.True(admission.LeaseActive);
        Assert.Equal(0, admission.Disposals);

        await enumerator.DisposeAsync();
        Assert.False(admission.LeaseActive);
        Assert.Equal(1, admission.Disposals);
    }

    [Fact]
    public async Task MalformedPartialStreamIsNotRestarted()
    {
        const string malformed = "data: {not-json}\n\n";
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.OK, malformed),
            _ => Response(HttpStatusCode.OK, SuccessfulResponse));
        using var client = Client(handler);
        var service = new OpenAIService(
            client,
            "key",
            transportOptions: new LLMTransportOptions());

        await Assert.ThrowsAsync<LLMTransportException>(async () =>
        {
            await foreach (var _ in service.StreamAsync(Request()))
            {
            }
        });

        Assert.Equal(1, handler.CallCount);
    }

    [Fact]
    public async Task UserCancellationIsSentOnceAndReleasesAdmissionLease()
    {
        var handler = new CancellationHandler();
        var admission = new RecordingAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            AdmissionPolicy = admission
        });
        using var cancellation = new CancellationTokenSource();

        var operation = service.GenerateAsync(Request(), cancellation.Token);
        await handler.Started.Task;
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => operation);
        Assert.Equal(1, handler.CallCount);
        Assert.Equal(1, admission.Disposals);
        Assert.False(admission.LeaseActive);
    }

    [Fact]
    public async Task TransientFailureSendsOneRequest()
    {
        var handler = new SequenceHandler(_ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"));
        using var client = Client(handler);
        var service = new OpenAIService(client, "key");
        await Assert.ThrowsAsync<LLMApiException>(() => service.GenerateAsync(Request()));
        Assert.Equal(1, handler.CallCount);
    }

    [Fact]
    public async Task AdmissionPolicyCanSerializeConcurrentLogicalRequests()
    {
        var handler = new FirstRequestBlockingHandler();
        var admission = new SerialAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            AdmissionPolicy = admission
        });

        var first = service.GenerateAsync(Request());
        await handler.FirstRequestStarted.Task;
        var second = service.GenerateAsync(Request());
        await admission.SecondAcquireStarted.Task;

        Assert.Equal(1, handler.CallCount);
        Assert.False(admission.SecondLeaseGranted.Task.IsCompleted);
        Assert.Equal(1, admission.MaximumActiveLeases);

        handler.ReleaseFirstRequest.TrySetResult();
        await Task.WhenAll(first, second);

        Assert.Equal(2, handler.CallCount);
        Assert.Equal(1, admission.MaximumActiveLeases);
        Assert.Equal(2, admission.Disposals);
    }

    [Fact]
    public async Task PerRequestTokenEstimateReachesAdmissionPolicy()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.OK, SuccessfulResponse));
        var admission = new RecordingAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            AdmissionPolicy = admission
        });
        var request = Request();
        request.Transport = new RequestTransportOptions
        {
            EstimatedInputTokens = 1234
        };

        var exception = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(request));

        Assert.Equal(1, handler.CallCount);
        Assert.Single(exception.Transport!.Attempts);
        Assert.Equal(1234, Assert.Single(admission.Contexts).EstimatedInputTokens);
    }

    private const string SuccessfulResponse = """
    {"id":"resp_1","model":"gpt-5.6","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}
    """;

    private static UnifiedRequest Request() => new()
    {
        Model = "gpt-5.6",
        Messages = { new UnifiedMessage(MessageRole.User, "Hello") }
    };

    private static HttpClient Client(HttpMessageHandler handler) => new(handler)
    {
        BaseAddress = new Uri("https://api.openai.com/v1/")
    };

    private static HttpResponseMessage Response(
        HttpStatusCode status,
        string body,
        params (string Name, string Value)[] headers)
    {
        var response = new HttpResponseMessage(status)
        {
            Content = new StringContent(body, Encoding.UTF8, "text/event-stream")
        };
        foreach (var header in headers)
            response.Headers.TryAddWithoutValidation(header.Name, header.Value);
        return response;
    }

    private sealed class SequenceHandler(params Func<HttpRequestMessage, HttpResponseMessage>[] responses)
        : HttpMessageHandler
    {
        public int CallCount { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            var index = CallCount++;
            var factory = responses[Math.Min(index, responses.Length - 1)];
            return Task.FromResult(factory(request));
        }
    }

    private sealed class RecordingAdmissionPolicy : IRequestAdmissionPolicy
    {
        public int Acquisitions { get; private set; }
        public int Disposals { get; private set; }
        public bool LeaseActive { get; private set; }
        public List<RequestAdmissionContext> Contexts { get; } = new();

        public ValueTask<IDisposable?> AcquireAsync(
            RequestAdmissionContext context,
            CancellationToken cancellationToken = default)
        {
            Acquisitions++;
            LeaseActive = true;
            Contexts.Add(context);
            return ValueTask.FromResult<IDisposable?>(new Lease(this));
        }

        private sealed class Lease(RecordingAdmissionPolicy owner) : IDisposable
        {
            public void Dispose()
            {
                owner.Disposals++;
                owner.LeaseActive = false;
            }
        }
    }

    private sealed class CancellationHandler : HttpMessageHandler
    {
        public int CallCount { get; private set; }
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            CallCount++;
            Started.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            throw new InvalidOperationException("Unreachable");
        }
    }

    private sealed class FirstRequestBlockingHandler : HttpMessageHandler
    {
        private int _callCount;
        public int CallCount => _callCount;
        public TaskCompletionSource FirstRequestStarted { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource ReleaseFirstRequest { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            var call = Interlocked.Increment(ref _callCount);
            if (call == 1)
            {
                FirstRequestStarted.TrySetResult();
                await ReleaseFirstRequest.Task.WaitAsync(cancellationToken);
            }
            return Response(HttpStatusCode.OK, SuccessfulResponse);
        }
    }

    private sealed class SerialAdmissionPolicy : IRequestAdmissionPolicy
    {
        private readonly SemaphoreSlim _semaphore = new(1, 1);
        private int _acquireCalls;
        private int _activeLeases;
        private int _disposals;

        public int MaximumActiveLeases { get; private set; }
        public int Disposals => _disposals;
        public TaskCompletionSource SecondAcquireStarted { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource SecondLeaseGranted { get; } =
            new(TaskCreationOptions.RunContinuationsAsynchronously);

        public async ValueTask<IDisposable?> AcquireAsync(
            RequestAdmissionContext context,
            CancellationToken cancellationToken = default)
        {
            var call = Interlocked.Increment(ref _acquireCalls);
            if (call == 2)
                SecondAcquireStarted.TrySetResult();
            await _semaphore.WaitAsync(cancellationToken);
            var active = Interlocked.Increment(ref _activeLeases);
            MaximumActiveLeases = Math.Max(MaximumActiveLeases, active);
            if (call == 2)
                SecondLeaseGranted.TrySetResult();
            return new Lease(this);
        }

        private sealed class Lease(SerialAdmissionPolicy owner) : IDisposable
        {
            private bool _disposed;

            public void Dispose()
            {
                if (_disposed)
                    return;
                _disposed = true;
                Interlocked.Decrement(ref owner._activeLeases);
                Interlocked.Increment(ref owner._disposals);
                owner._semaphore.Release();
            }
        }
    }
}
