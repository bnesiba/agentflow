# LLM Abstraction Layer for C#

A unified C# abstraction layer for interacting with multiple Large Language Model (LLM) APIs including OpenAI, Google Gemini, and Anthropic Claude. This library provides a consistent interface across providers, making it easy to switch between different LLM services or support multiple providers in your application.

## Features

- **Unified Interface**: Single API for OpenAI, Gemini, and Claude
- **Type-Safe Models**: Strongly-typed request and response models
- **Provider Converters**: Automatic conversion between unified and provider-specific formats
- **Structured Output**: JSON mode and JSON Schema support across all providers
- **Streaming**: Real-time streaming responses with unified chunk format
- **Tool/Function Calling**: Consistent tool calling interface across all providers
- **Multimodal Support**: Text and image inputs with unified content blocks
- **Async/Await**: Modern async/await patterns throughout
- **Extensible**: Easy to add new providers or extend existing functionality

## Installation

```bash
# Add the project to your solution
dotnet add reference path/to/LLMAbstraction.csproj
```

## Quick Start

### Basic Text Generation

```csharp
using LLMAbstraction.Core;
using LLMAbstraction.Core.Models;

// Create a service (OpenAI, Claude, or Gemini)
var service = LLMServiceFactory.CreateOpenAI("your-api-key");

// Create a request
var request = new UnifiedRequest
{
    Model = "gpt-5-mini",
    Messages = new List<UnifiedMessage>
    {
        new UnifiedMessage(MessageRole.User, "What is the capital of France?")
    },
    Parameters = new GenerationParameters
    {
        MaxOutputTokens = 100,
        Temperature = 0.7
    }
};

// Generate response
var response = await service.GenerateAsync(request);
var text = (response.Choices[0].Message.Content[0] as TextContent)?.Text;
Console.WriteLine(text);
```

### Using Different Providers

The same code works with any provider - just change the service creation:

```csharp
// OpenAI
var openAI = LLMServiceFactory.CreateOpenAI("openai-key");
request.Model = "gpt-5-mini";

// Claude
var claude = LLMServiceFactory.CreateClaude("claude-key");
request.Model = "claude-sonnet-4-20250514";

// Gemini
var gemini = LLMServiceFactory.CreateGemini("gemini-key");
request.Model = "gemini-3-pro";

// All use the same GenerateAsync method
var response = await service.GenerateAsync(request);
```

## Core Concepts

### Unified Message Structure

Messages use a role-based structure with flexible content blocks:

```csharp
var message = new UnifiedMessage
{
    Role = MessageRole.User,
    Content = new List<ContentBlock>
    {
        new TextContent { Text = "Hello!" },
        new ImageContent 
        { 
            Source = new ImageSource { Url = "https://..." }
        }
    }
};
```

**Supported Roles:**
- `System`: System instructions (handled appropriately per provider)
- `User`: User messages
- `Assistant`: Assistant/model responses
- `Tool`: Tool execution results

**Content Block Types:**
- `TextContent`: Plain text
- `ImageContent`: Images (URL or base64)
- `ToolCallContent`: Tool/function calls from the model
- `ToolResultContent`: Results from tool execution

### Generation Parameters

Common parameters across all providers:

```csharp
var parameters = new GenerationParameters
{
    MaxOutputTokens = 1000,           // Maximum tokens to generate
    Temperature = 0.7,          // Sampling temperature (0.0-2.0)
    TopP = 0.9,                 // Nucleus sampling
    TopK = 40,                  // Top-k sampling (Gemini/Claude)
    StopSequences = new List<string> { "\n\n" },
    Stream = false              // For future streaming support
};
```

### Tool/Function Calling

Define tools with JSON Schema:

```csharp
var tool = new ToolDefinition
{
    Name = "get_weather",
    Description = "Get current weather for a location",
    Parameters = new Dictionary<string, object>
    {
        ["type"] = "object",
        ["properties"] = new Dictionary<string, object>
        {
            ["location"] = new Dictionary<string, object>
            {
                ["type"] = "string",
                ["description"] = "City and state, e.g. Boston, MA"
            }
        },
        ["required"] = new[] { "location" }
    }
};

var request = new UnifiedRequest
{
    Model = "gpt-5-mini",
    Messages = messages,
    Tools = new List<ToolDefinition> { tool },
    ToolChoice = new ToolChoice { Type = ToolChoiceType.Auto }
};
```

**Tool Choice Types:**
- `Auto`: Model decides whether to use tools
- `None`: Don't use tools
- `Required`: Must use at least one tool
- `Specific`: Use a specific tool (set `ToolName`)

## Provider-Specific Details

### OpenAI

```csharp
var service = LLMServiceFactory.CreateOpenAI(
    apiKey: "sk-...",
    baseUrl: "https://api.openai.com/v1"  // Optional
);
```

**Supported Models:** `gpt-5-mini`, `gpt-5.1`, `gpt-5.2`, etc.

**Notes:**
- Supports 6 message roles (system, user, assistant, tool, function, developer)
- System messages are included in the messages array
- Tool calls use dedicated message types

### Claude (Anthropic)

```csharp
var service = LLMServiceFactory.CreateClaude(
    apiKey: "sk-ant-...",
    apiVersion: "2023-06-01",              // Optional
    baseUrl: "https://api.anthropic.com"   // Optional
);
```

**Supported Models:** `claude-opus-4-20250514`, `claude-sonnet-4-20250514`, `claude-haiku-3-5-20241022`

**Notes:**
- Uses only 2 roles (user, assistant)
- System instructions are a separate parameter
- `max_tokens` is required (defaults to 1024)
- Supports context caching with cache metrics in response

### Gemini (Google)

```csharp
var service = LLMServiceFactory.CreateGemini(
    apiKey: "AIza...",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta"  // Optional
);
```

**Supported Models:** `gemini-3-pro`, `gemini-2.5-flash`, `gemini-1.5-pro`

**Notes:**
- Uses only 2 roles (user, model)
- Model name is included in the URL path
- System instructions are a separate parameter
- Part-based content structure (handled by converter)
- Includes safety ratings in responses

## Architecture

### Project Structure

```
LLMAbstraction/
|-- Core/
|   |-- Models/
|   |   |-- UnifiedMessage.cs       # Message and content block models
|   |   |-- UnifiedRequest.cs       # Request model
|   |   `-- UnifiedResponse.cs      # Response model
|   |-- Interfaces/
|   |   `-- ILLMService.cs          # Core service interface
|   `-- LLMServiceFactory.cs        # Factory for creating services
|-- Providers/
|   |-- OpenAI/
|   |   |-- Models/
|   |   |   `-- OpenAIModels.cs     # OpenAI-specific models
|   |   |-- OpenAIConverter.cs      # Conversion logic
|   |   `-- OpenAIService.cs        # Service implementation
|   |-- Claude/
|   |   |-- Models/
|   |   |   `-- ClaudeModels.cs
|   |   |-- ClaudeConverter.cs
|   |   `-- ClaudeService.cs
|   `-- Gemini/
|       |-- Models/
|       |   `-- GeminiModels.cs
|       |-- GeminiConverter.cs
|       `-- GeminiService.cs
`-- Examples/
    `-- BasicUsage.cs               # Usage examples
```

### Design Patterns

**Converter Pattern**: Each provider has a converter that implements `IModelConverter<TRequest, TResponse>` to translate between unified and provider-specific formats.

**Factory Pattern**: `LLMServiceFactory` provides a centralized way to create service instances.

**Strategy Pattern**: Different providers implement the same `ILLMService` interface, allowing runtime provider selection.

## Advanced Usage

### Multi-Turn Conversations

```csharp
var request = new UnifiedRequest
{
    Model = "claude-sonnet-4-20250514",
    Instructions = "You are a helpful coding assistant.",
    Messages = new List<UnifiedMessage>
    {
        new UnifiedMessage(MessageRole.User, "How do I reverse a string in C#?"),
        new UnifiedMessage(MessageRole.Assistant, "You can use Array.Reverse()..."),
        new UnifiedMessage(MessageRole.User, "Can you show me an example?")
    },
    Parameters = new GenerationParameters { MaxOutputTokens = 500 }
};
```

### Handling Tool Calls

```csharp
var response = await service.GenerateAsync(request);

if (response.Choices[0].FinishReason == FinishReason.ToolCalls)
{
    foreach (var content in response.Choices[0].Message.Content)
    {
        if (content is ToolCallContent toolCall)
        {
            // Execute the tool
            var result = ExecuteTool(toolCall.Name, toolCall.Input);
            
            // Add tool result to conversation
            request.Messages.Add(response.Choices[0].Message);
            request.Messages.Add(new UnifiedMessage(
                MessageRole.Tool,
                new List<ContentBlock>
                {
                    new ToolResultContent
                    {
                        ToolCallId = toolCall.Id,
                        Output = result
                    }
                }
            ));
            
            // Continue conversation
            var finalResponse = await service.GenerateAsync(request);
        }
    }
}
```

### Provider-Specific Options

Use the `ProviderOptions` escape hatch for provider-specific features:

```csharp
var request = new UnifiedRequest
{
    Model = "gpt-5-mini",
    Messages = messages,
    Parameters = parameters,
    ProviderOptions = new ProviderOptions
    {
        OpenAI = new Dictionary<string, object>
        {
            ["presence_penalty"] = 0.5,
            ["frequency_penalty"] = 0.3
        }
    }
};
```

## Structured Output

All providers support structured output with JSON mode and JSON Schema:

### JSON Mode

```csharp
var request = new UnifiedRequest
{
    Model = "gpt-5-mini",
    Messages = new List<UnifiedMessage>
    {
        new UnifiedMessage(MessageRole.User, "List 3 colors with hex codes")
    },
    ResponseFormat = new ResponseFormat
    {
        Type = ResponseFormatType.Json
    }
};

var response = await service.GenerateAsync(request);
// Response will be valid JSON
```

### JSON Schema

Define a strict schema for the response:

```csharp
var personSchema = new Dictionary<string, object>
{
    ["type"] = "object",
    ["properties"] = new Dictionary<string, object>
    {
        ["name"] = new Dictionary<string, object> { ["type"] = "string" },
        ["age"] = new Dictionary<string, object> { ["type"] = "number" },
        ["email"] = new Dictionary<string, object> { ["type"] = "string" }
    },
    ["required"] = new[] { "name", "age", "email" }
};

var request = new UnifiedRequest
{
    Model = "gpt-5-mini",
    Messages = new List<UnifiedMessage>
    {
        new UnifiedMessage(MessageRole.User, "Generate a person profile")
    },
    ResponseFormat = new ResponseFormat
    {
        Type = ResponseFormatType.JsonSchema,
        JsonSchema = new JsonSchema
        {
            Name = "person",
            Schema = personSchema,
            Strict = true  // Enforce strict validation
        }
    }
};
```

**Provider Notes:**
- **OpenAI**: Native `response_format` support
- **Claude**: Native `output_config.format` support (available since late 2024)
- **Gemini**: Uses `responseMimeType` and `responseJsonSchema` in generation config

## Streaming

All providers support real-time streaming responses:

### Basic Streaming

```csharp
var request = new UnifiedRequest
{
    Model = "gpt-5-mini",
    Messages = new List<UnifiedMessage>
    {
        new UnifiedMessage(MessageRole.User, "Write a poem about coding")
    }
};

await foreach (var chunk in service.StreamAsync(request))
{
    if (!string.IsNullOrEmpty(chunk.Delta.Content))
    {
        Console.Write(chunk.Delta.Content);
    }

    if (chunk.FinishReason.HasValue)
    {
        Console.WriteLine($"\nFinished: {chunk.FinishReason}");
        if (chunk.Usage != null)
        {
            Console.WriteLine($"Tokens: {chunk.Usage.TotalTokens}");
        }
    }
}
```

### Streaming with Aggregation

```csharp
var fullResponse = new StringBuilder();

await foreach (var chunk in service.StreamAsync(request))
{
    if (!string.IsNullOrEmpty(chunk.Delta.Content))
    {
        fullResponse.Append(chunk.Delta.Content);
    }
}

Console.WriteLine(fullResponse.ToString());
```

### Streaming Tool Calls

```csharp
var toolCallArgs = new StringBuilder();

await foreach (var chunk in service.StreamAsync(request))
{
    if (chunk.Delta.ToolCalls != null)
    {
        foreach (var toolCall in chunk.Delta.ToolCalls)
        {
            if (!string.IsNullOrEmpty(toolCall.Arguments))
            {
                toolCallArgs.Append(toolCall.Arguments);
            }
        }
    }
}
```

**Stream Chunk Properties:**
- `Id`: Unique identifier for the stream
- `Model`: Model that generated the chunk
- `ChoiceIndex`: Index for multiple choices (n > 1)
- `Delta`: Incremental content (text, tool calls)
- `FinishReason`: Reason for completion (only in final chunk)
- `Usage`: Token usage (only in final chunk for some providers)

## Future Enhancements

- Response caching
- Retry logic with exponential backoff
- Rate limiting and throttling
- Token counting utilities
- Batch processing support
- Conversation history management

## Error Handling

All services throw `HttpRequestException` for API errors:

```csharp
try
{
    var response = await service.GenerateAsync(request);
}
catch (HttpRequestException ex)
{
    Console.WriteLine($"API Error: {ex.Message}");
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"Deserialization Error: {ex.Message}");
}
```

## Dependencies

- .NET 8.0 or higher

## Contributing

Contributions are welcome! Areas for contribution:
- Additional provider implementations
- Streaming support
- Enhanced error handling
- Unit tests
- Performance optimizations

## License

MIT License - see LICENSE file for details

## Acknowledgments

This abstraction layer is based on analysis of the official API documentation from:
- OpenAI (https://platform.openai.com/docs)
- Anthropic Claude (https://docs.anthropic.com)
- Google Gemini (https://ai.google.dev/docs)
