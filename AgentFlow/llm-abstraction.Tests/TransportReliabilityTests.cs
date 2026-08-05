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
    public async Task TransientStatusIsRetriedWithServerDelayAndAttemptMetadata()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.TooManyRequests, """{"error":{"message":"slow down"}}""",
                ("retry-after", "5")),
            _ => Response(HttpStatusCode.OK, SuccessfulResponse,
                ("x-request-id", "req_success"),
                ("x-ratelimit-limit-requests", "100"),
                ("x-ratelimit-remaining-requests", "99"),
                ("x-ratelimit-reset-requests", "2s")));
        var delay = new RecordingDelayScheduler();
        var observer = new RecordingRetryObserver();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: Options(delay, observer));

        var response = await service.GenerateAsync(Request());

        Assert.Equal(2, handler.CallCount);
        Assert.Equal(TimeSpan.FromSeconds(5), Assert.Single(delay.Delays));
        Assert.Equal(TimeSpan.FromSeconds(5), Assert.Single(observer.Contexts).Attempt.ServerDelay);
        Assert.Equal("req_success", response.Transport!.RequestId);
        Assert.Collection(response.Transport.Attempts,
            first =>
            {
                Assert.Equal(HttpStatusCode.TooManyRequests, first.StatusCode);
                Assert.True(first.WillRetry);
            },
            second =>
            {
                Assert.Equal(HttpStatusCode.OK, second.StatusCode);
                Assert.False(second.WillRetry);
            });
        Assert.Equal(100, response.Transport.RateLimits!.Requests!.Limit);
        Assert.Equal(99, response.Transport.RateLimits.Requests.Remaining);
        Assert.Equal(TimeSpan.FromSeconds(2), response.Transport.RateLimits.Requests.ResetsAfter);
    }

    [Fact]
    public async Task NonTransientStatusIsNeverRetried()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.BadRequest,
                """{"error":{"message":"bad request","type":"invalid_request_error"}}"""));
        var delay = new RecordingDelayScheduler();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: Options(delay));

        var exception = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(Request()));

        Assert.Equal(1, handler.CallCount);
        Assert.Empty(delay.Delays);
        Assert.Single(exception.Transport!.Attempts);
        Assert.False(exception.Transport.Attempts[0].WillRetry);
    }

    [Fact]
    public async Task ExhaustedTransientResponsesRetainEveryAttempt()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"));
        var delay = new RecordingDelayScheduler();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: Options(delay));

        var exception = await Assert.ThrowsAsync<LLMApiException>(
            () => service.GenerateAsync(Request()));

        Assert.Equal(3, handler.CallCount);
        Assert.Equal(2, delay.Delays.Count);
        Assert.Equal(3, exception.Transport!.Attempts.Count);
        Assert.True(exception.Transport.Attempts[0].WillRetry);
        Assert.True(exception.Transport.Attempts[1].WillRetry);
        Assert.False(exception.Transport.Attempts[2].WillRetry);
    }

    [Fact]
    public async Task ConnectionFailureRetriesAndProducesStructuredTransportFailure()
    {
        var handler = new SequenceHandler(
            _ => throw new HttpRequestException("connection reset"),
            _ => throw new HttpRequestException("connection reset"),
            _ => throw new HttpRequestException("connection reset"));
        var delay = new RecordingDelayScheduler();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: Options(delay));

        var exception = await Assert.ThrowsAsync<LLMTransportException>(
            () => service.GenerateAsync(Request()));

        Assert.Equal(LLMProvider.OpenAI, exception.Provider);
        Assert.True(exception.IsTransient);
        Assert.Equal(3, exception.Transport.Attempts.Count);
        Assert.All(exception.Transport.Attempts,
            attempt => Assert.Contains(nameof(HttpRequestException), attempt.ExceptionType));
    }

    [Fact]
    public async Task StreamRetriesInitialStatusAndAttachesMetadataToFirstChunk()
    {
        var completed = """
        data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5.6","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"Hi"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}

        """;
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.BadGateway, "bad gateway"),
            _ => Response(HttpStatusCode.OK, completed));
        var delay = new RecordingDelayScheduler();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: Options(delay));

        var chunks = new List<StreamChunk>();
        await foreach (var chunk in service.StreamAsync(Request()))
            chunks.Add(chunk);

        var first = Assert.Single(chunks);
        Assert.Equal(2, handler.CallCount);
        Assert.Equal(2, first.Transport!.Attempts.Count);
        Assert.NotNull(first.CompletedMessage);
    }

    [Fact]
    public async Task AdmissionLeaseCoversRetriesAndResponseProcessing()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.OK, SuccessfulResponse));
        var admission = new RecordingAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            Retry = ZeroDelayPolicy(),
            DelayScheduler = new RecordingDelayScheduler(),
            AdmissionPolicy = admission
        });

        Assert.False(admission.LeaseActive);
        await service.GenerateAsync(Request());

        Assert.Equal(1, admission.Acquisitions);
        Assert.Equal(1, admission.Disposals);
        Assert.False(admission.LeaseActive);
        Assert.Equal(2, handler.CallCount);
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
            Retry = ZeroDelayPolicy(),
            DelayScheduler = new RecordingDelayScheduler(),
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
            transportOptions: Options(new RecordingDelayScheduler()));

        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in service.StreamAsync(Request()))
            {
            }
        });

        Assert.Equal(1, handler.CallCount);
    }

    [Fact]
    public async Task UserCancellationIsNotRetriedAndReleasesAdmissionLease()
    {
        var handler = new CancellationHandler();
        var admission = new RecordingAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            Retry = ZeroDelayPolicy(),
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
    public async Task CancellationDuringBackoffStopsBeforeAnotherAttempt()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.OK, SuccessfulResponse));
        var delay = new BlockingDelayScheduler();
        var admission = new RecordingAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            Retry = ZeroDelayPolicy(),
            DelayScheduler = delay,
            AdmissionPolicy = admission
        });
        using var cancellation = new CancellationTokenSource();

        var operation = service.GenerateAsync(Request(), cancellation.Token);
        await delay.Started.Task;
        cancellation.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => operation);
        Assert.Equal(1, handler.CallCount);
        Assert.Equal(1, admission.Disposals);
    }

    [Fact]
    public async Task AdmissionPolicyCanSerializeConcurrentLogicalRequests()
    {
        var handler = new FirstRequestBlockingHandler();
        var admission = new SerialAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            Retry = ZeroDelayPolicy(),
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
    public async Task RetryObserverFailureIsNotMisclassifiedAsAProviderFailure()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.OK, SuccessfulResponse));
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            Retry = ZeroDelayPolicy(),
            RetryObserver = new ThrowingRetryObserver()
        });

        var exception = await Assert.ThrowsAsync<HttpRequestException>(
            () => service.GenerateAsync(Request()));

        Assert.Equal("observer failed", exception.Message);
        Assert.Equal(1, handler.CallCount);
    }

    [Fact]
    public async Task PerRequestRetryOverrideAndTokenEstimateReachTheTransportLayer()
    {
        var handler = new SequenceHandler(
            _ => Response(HttpStatusCode.ServiceUnavailable, "unavailable"),
            _ => Response(HttpStatusCode.OK, SuccessfulResponse));
        var admission = new RecordingAdmissionPolicy();
        using var client = Client(handler);
        var service = new OpenAIService(client, "key", transportOptions: new LLMTransportOptions
        {
            Retry = ZeroDelayPolicy(),
            AdmissionPolicy = admission
        });
        var request = Request();
        request.Transport = new RequestTransportOptions
        {
            Retry = new RetryPolicy { Enabled = false },
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

    private static LLMTransportOptions Options(
        RecordingDelayScheduler delay,
        RecordingRetryObserver? observer = null) => new()
    {
        Retry = ZeroDelayPolicy(),
        DelayScheduler = delay,
        RetryObserver = observer,
        JitterSource = () => 0
    };

    private static RetryPolicy ZeroDelayPolicy() => new()
    {
        MaximumAttempts = 3,
        BaseDelay = TimeSpan.Zero,
        MaximumDelay = TimeSpan.Zero,
        MaximumElapsedTime = TimeSpan.FromMinutes(1),
        UseJitter = false
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

    private sealed class RecordingDelayScheduler : IRetryDelayScheduler
    {
        public List<TimeSpan> Delays { get; } = new();
        public ValueTask DelayAsync(TimeSpan delay, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Delays.Add(delay);
            return ValueTask.CompletedTask;
        }
    }

    private sealed class BlockingDelayScheduler : IRetryDelayScheduler
    {
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public async ValueTask DelayAsync(TimeSpan delay, CancellationToken cancellationToken = default)
        {
            Started.TrySetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
        }
    }

    private sealed class RecordingRetryObserver : IRetryObserver
    {
        public List<RetryContext> Contexts { get; } = new();
        public ValueTask OnRetryAsync(RetryContext context, CancellationToken cancellationToken = default)
        {
            Contexts.Add(context);
            return ValueTask.CompletedTask;
        }
    }

    private sealed class ThrowingRetryObserver : IRetryObserver
    {
        public ValueTask OnRetryAsync(RetryContext context, CancellationToken cancellationToken = default) =>
            ValueTask.FromException(new HttpRequestException("observer failed"));
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
