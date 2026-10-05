using System;
using System.Net.Http;
using LLMAbstraction.Core.Transport;

namespace LLMAbstraction.Core.Errors
{
    public sealed class LLMTransportException : HttpRequestException
    {
        public LLMTransportException(
            LLMProvider provider,
            string message,
            bool isTransient,
            TransportMetadata transport,
            Exception innerException)
            : base(message, innerException)
        {
            Provider = provider;
            IsTransient = isTransient;
            Transport = transport;
        }

        public LLMProvider Provider { get; }
        public bool IsTransient { get; }
        public TransportMetadata Transport { get; }
    }
}
