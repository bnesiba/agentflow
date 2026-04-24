# LLM Abstraction Layer - Test Suite

This directory contains comprehensive tests for the LLM abstraction layer, covering unit tests, integration tests, and end-to-end tests.

## Test Structure

```
Tests/
├── Core/
│   └── UnifiedModelsTests.cs          # Tests for core unified models
├── Converters/
│   ├── OpenAIConverterTests.cs        # OpenAI converter unit tests
│   ├── ClaudeConverterTests.cs        # Claude converter unit tests
│   └── GeminiConverterTests.cs        # Gemini converter unit tests
├── Services/
│   ├── OpenAIServiceTests.cs          # OpenAI service tests with HTTP mocking
│   ├── ClaudeServiceTests.cs          # Claude service tests with HTTP mocking
│   └── GeminiServiceTests.cs          # Gemini service tests with HTTP mocking
├── Integration/
│   ├── LLMServiceFactoryTests.cs      # Factory pattern tests
│   └── EndToEndTests.cs               # E2E tests (require real API keys)
├── Helpers/
│   └── TestHelpers.cs                 # Test data builders and utilities
└── LLMAbstraction.Tests.csproj        # Test project file
```

## Running Tests

### Run All Tests (Except E2E)

```bash
dotnet test
```

### Run Specific Test Category

```bash
# Run only converter tests
dotnet test --filter FullyQualifiedName~Converters

# Run only service tests
dotnet test --filter FullyQualifiedName~Services

# Run only integration tests
dotnet test --filter FullyQualifiedName~Integration
```

### Run Tests with Coverage

```bash
dotnet test /p:CollectCoverage=true /p:CoverletOutputFormat=opencover
```

## Test Categories

### 1. Core Model Tests (`Core/`)

Tests for the unified data models:
- Message construction
- Content block types
- Parameter validation
- Model defaults

**Coverage:**
- UnifiedMessage
- UnifiedRequest
- UnifiedResponse
- ContentBlock types (Text, Image, ToolCall, ToolResult)
- GenerationParameters
- ToolDefinition

### 2. Converter Tests (`Converters/`)

Unit tests for provider-specific converters:
- Request conversion (unified → provider)
- Response conversion (provider → unified)
- Edge cases and special features
- Parameter mapping

**Test Cases per Provider:**
- Simple text messages
- System message handling
- Multi-turn conversations
- Multimodal content (images)
- Tool/function calling
- Tool choice strategies
- Finish reason mapping
- Token usage tracking

**Providers Covered:**
- OpenAI (OpenAIConverter)
- Claude (ClaudeConverter)
- Gemini (GeminiConverter)

### 3. Service Tests (`Services/`)

HTTP service tests with mocked responses:
- API request formatting
- Response parsing
- Error handling
- Authentication headers
- URL construction

**Test Cases per Provider:**
- Simple generation
- System message inclusion
- Tool calling
- API errors (4xx, 5xx)
- Authentication validation
- Custom base URLs

**Mocking:**
Uses `Moq` to mock `HttpMessageHandler` for HTTP testing

### 4. Integration Tests (`Integration/`)

Higher-level tests for component integration:
- Factory pattern validation
- Service creation
- Configuration handling

**LLMServiceFactoryTests:**
- Create services for all providers
- Configuration-based creation
- API key validation
- Custom settings

**EndToEndTests (Skipped by Default):**
- Real API calls (requires valid API keys)
- Cross-provider consistency
- Multi-turn conversations
- System instruction adherence

## End-to-End Tests

E2E tests are **skipped by default** because they require real API keys and make actual API calls.

### Running E2E Tests

1. Set environment variables:
```bash
export OPENAI_API_KEY="your-openai-key"
export CLAUDE_API_KEY="your-claude-key"
export GEMINI_API_KEY="your-gemini-key"
```

2. Run tests without skip filter:
```bash
dotnet test --filter FullyQualifiedName~EndToEndTests
```

Or modify the test file to remove `Skip = "..."` from the `[Fact]` attributes.

### E2E Test Coverage

- Simple text generation (all providers)
- Cross-provider consistency
- System message handling
- Multi-turn conversation context
- Response format validation

**Note:** E2E tests consume API credits and should be run sparingly.

## Test Dependencies

The test project uses minimal dependencies:

- **xUnit** (2.6.2) - Test framework
- **Moq** (4.20.70) - Mocking framework for HTTP message handlers
- **Microsoft.NET.Test.Sdk** (17.8.0) - Test SDK

All assertions use xUnit's built-in `Assert` class - no additional assertion libraries needed.

## Writing New Tests

### Test Helper Usage

Use `TestHelpers` to create common test data:

```csharp
// Simple request
var request = TestHelpers.CreateSimpleRequest("gpt-4o");

// Request with system message
var request = TestHelpers.CreateRequestWithSystem("You are helpful.");

// Multi-turn conversation
var request = TestHelpers.CreateMultiTurnRequest();

// Multimodal request
var request = TestHelpers.CreateMultimodalRequest();

// Tool calling request
var request = TestHelpers.CreateToolCallRequest();
```

### Converter Test Pattern

```csharp
[Fact]
public void ConvertRequest_YourScenario_ExpectedBehavior()
{
    // Arrange
    var converter = new YourConverter();
    var request = TestHelpers.CreateSimpleRequest();

    // Act
    var result = converter.ConvertRequest(request);

    // Assert
    result.Should().NotBeNull();
    result.SomeProperty.Should().Be(expectedValue);
}
```

### Service Test Pattern

```csharp
[Fact]
public async Task GenerateAsync_YourScenario_ExpectedBehavior()
{
    // Arrange
    var mockHttp = new MockHttpMessageHandler();
    mockHttp.When(HttpMethod.Post, "url")
        .Respond("application/json", mockResponseJson);
    
    var httpClient = mockHttp.ToHttpClient();
    var service = new YourService(httpClient, "api-key");
    var request = TestHelpers.CreateSimpleRequest();

    // Act
    var result = await service.GenerateAsync(request);

    // Assert
    result.Should().NotBeNull();
}
```

## Test Coverage Goals

- **Line Coverage:** > 80%
- **Branch Coverage:** > 75%
- **Converter Coverage:** 100% (all conversion paths)
- **Service Coverage:** > 90% (excluding E2E)

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
- Fast execution (< 5 seconds for unit tests)
- No external dependencies (except E2E)
- Deterministic results
- Clear failure messages

## Troubleshooting

### Tests Failing Locally

1. Restore NuGet packages:
```bash
dotnet restore
```

2. Clean and rebuild:
```bash
dotnet clean
dotnet build
```

3. Run tests with verbose output:
```bash
dotnet test --logger "console;verbosity=detailed"
```

### HTTP Mocking Issues

If HTTP mocking tests fail:
- Verify `Moq` is installed
- Check mock setup uses `Protected()` for `HttpMessageHandler`
- Ensure JSON serialization options match

### E2E Test Failures

If E2E tests fail:
- Verify API keys are valid and have credits
- Check model names are current
- Review API rate limits
- Ensure network connectivity

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add integration tests for new providers
4. Update this README with new test categories
5. Maintain > 80% code coverage

## License

Same as parent project (MIT)
