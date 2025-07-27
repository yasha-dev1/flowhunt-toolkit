# Flowhunt Toolkit - Development Guide

## Overview

This is a CLI tool for FlowHunt Flow Engineers to streamline the process of developing flows in FlowHunt for advanced users.

## Project Structure

```
flowhunt-dev-toolkit/
├── flowhunt_toolkit/           # Main package
│   ├── __init__.py            # Package initialization
│   ├── cli.py                 # CLI commands and interface
│   └── core/                  # Core functionality modules
│       ├── __init__.py
│       ├── client.py          # FlowHunt API client wrapper
│       └── evaluator.py       # Flow evaluation logic
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_cli.py           # CLI tests
│   └── test_evaluator.py     # Evaluator tests
├── .github/workflows/         # CI/CD pipelines
│   ├── test.yml              # Testing workflow
│   └── release.yml           # Release workflow
├── docs/                     # Documentation
├── pyproject.toml           # Python packaging and dependencies
└── README.md                # Project documentation
```

## Development Setup

1. Clone the repository
2. Install in development mode:
   ```bash
   pip install -e .
   ```
3. Install development dependencies:
   ```bash
   pip install pytest pytest-cov
   ```

## Running Tests

```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=flowhunt_toolkit --cov-report=html
```

## CLI Commands

### Available Commands

- `flowhunt evaluate` - Evaluate a flow using LLM as a judge
- `flowhunt inspect` - Inspect a FlowHunt flow's configuration
- `flowhunt batch-run` - Run a flow in batch mode
- `flowhunt auth` - Authenticate with FlowHunt API
- `flowhunt list-flows` - List available flows

### Example Usage

```bash
# Evaluate a flow
flowhunt evaluate questions.csv flow-id-123 --output results.json

# Inspect a flow
flowhunt inspect flow-id-123 --format json

# Batch run
flowhunt batch-run inputs.csv flow-id-123 --parallel 5
```

## Implementation Status

### ✅ Completed
- CLI boilerplate with Click
- Project structure and packaging
- Test suite framework
- CI/CD pipeline setup
- Core module stubs

### 🚧 In Progress
- FlowHunt SDK integration
- LLM as a judge evaluation logic
- Authentication system

### 📋 TODO
- Implement actual FlowHunt API calls
- Add configuration file support
- Create Homebrew formula
- Create Debian package
- Add more comprehensive tests

## Contributing

1. Make sure all tests pass
2. Add tests for new functionality
3. Update documentation as needed
4. Follow Python PEP 8 style guidelines

## Packaging and Distribution

The project is set up for distribution via:
- PyPI (Python Package Index)
- Homebrew (macOS package manager)
- APT (Debian/Ubuntu package manager)

Build commands:
```bash
python -m build
twine upload dist/*
```
