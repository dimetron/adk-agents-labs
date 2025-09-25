# Testing Guide for ADK Agents Labs

## Overview

This project now includes comprehensive Python tests covering all major components:

- **44 test cases** across unit and integration tests
- **Full coverage** of OllamaLlm class functionality
- **Agent workflow testing** for both parent and workflow agents
- **Mock fixtures** for reliable, fast testing
- **Makefile targets** for easy test execution

## Quick Start

```bash
# Install dependencies (including test deps)
make install

# Run all tests
make test

# Run with coverage report
make test-coverage
```

## What's Tested

### Unit Tests (20 tests)
- **OllamaLlm class** (18 tests)
  - Initialization and configuration
  - Model name normalization
  - Message conversion
  - Response parsing
  - Error handling
  - HTTP client interactions
  
- **Callback logging** (9 tests)
  - Query logging functionality
  - Response logging with different content types
  - Function call logging

### Integration Tests (15 tests)
- **Parent agents** (7 tests)
  - Travel planning workflow
  - State management
  - Agent configuration
  - Model resolution
  
- **Workflow agents** (9 tests)
  - Movie writing workflow
  - File operations
  - Wikipedia integration
  - Complex state management
  - Loop agent behavior

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_helpers.py          # Test utilities and mock classes
├── unit/                    # Unit tests
│   ├── test_ollama_llm.py
│   └── test_callback_logging.py
├── integration/             # Integration tests
│   ├── test_parent_agents.py
│   └── test_workflow_agents.py
└── fixtures/                # Mock data and responses
    └── mock_responses.py
```

## Key Features

### Comprehensive Mocking
- HTTP client mocking for Ollama API calls
- Environment variable management
- File system operations
- Agent state management

### Async Testing Support
- Proper async/await test handling
- Streaming response testing
- Generator function testing

### Error Scenario Coverage
- HTTP error handling
- Model not found scenarios
- Invalid configuration testing
- Network failure simulation

### Real-world Scenarios
- Complete agent workflows
- Tool function execution
- State persistence across agent calls
- Complex multi-agent interactions

## Running Tests

### All Tests
```bash
make test                    # Run all tests
make test-unit              # Run only unit tests
make test-integration       # Run only integration tests
make test-coverage          # Run with coverage report
```

### Specific Tests
```bash
# Run specific test file
.venv/bin/pytest tests/unit/test_ollama_llm.py

# Run specific test method
.venv/bin/pytest tests/unit/test_ollama_llm.py::TestOllamaLlm::test_init_with_defaults

# Run with verbose output
.venv/bin/pytest -v
```

### Coverage Reports
After running `make test-coverage`, open `htmlcov/index.html` to see detailed coverage information.

## CI/CD Ready

Tests are designed to run in CI environments:
- No external dependencies (Ollama is mocked)
- Isolated test execution
- Comprehensive error handling
- Fast execution (< 10 seconds for full suite)

## Next Steps

The test suite provides a solid foundation for:
1. **Regression testing** - Catch breaking changes early
2. **Refactoring confidence** - Modify code safely
3. **Documentation** - Tests serve as usage examples
4. **Quality assurance** - Maintain code quality standards

## Troubleshooting

If tests fail:
1. Check that virtual environment is activated
2. Ensure all dependencies are installed (`make install`)
3. Run with verbose output (`pytest -v`) for detailed error info
4. Check the test logs for specific failure reasons

For more detailed information, see `tests/README.md`.

