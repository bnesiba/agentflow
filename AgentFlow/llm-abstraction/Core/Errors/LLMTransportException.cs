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

        /// <summary>True when stream content was delivered before this failure.</summary>
        public bool HasPartialOutput { get; internal set; }
        public bool HasToolCalls { get; internal set; }
        public LLMProvider Provider { get; }
        public bool IsTransient { get; }
        public TransportMetadata Transport { get; }
    }
}
