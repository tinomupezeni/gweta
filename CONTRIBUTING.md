# Contributing to Gweta

Thank you for your interest in contributing to Gweta! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/tinomupezeni/gweta.git
   cd gweta
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style

We use the following tools for code quality:

- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Pytest** for testing

Run checks locally:
```bash
# Linting
ruff check src/gweta tests

# Formatting
ruff format src/gweta tests

# Type checking
mypy src/gweta

# Tests
pytest
```

## Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following these guidelines:
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. Run the test suite:
   ```bash
   pytest --cov=gweta
   ```

4. Push your changes and create a pull request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure CI checks pass
- Request review from maintainers

## Project Structure

```
gweta/
├── src/gweta/
│   ├── acquire/      # Data acquisition (crawling, PDF, API, DB)
│   ├── validate/     # Validation pipeline
│   ├── ingest/       # Chunking and vector store ingestion
│   ├── mcp/          # MCP server implementation
│   ├── adapters/     # Framework adapters (LangChain, etc.)
│   ├── cli/          # Command-line interface
│   └── core/         # Core types and utilities
├── tests/
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
└── docs/             # Documentation
```

## Adding New Features

### New Validator
1. Create detector in `src/gweta/validate/detectors/`
2. Integrate with `ChunkValidator` in `chunks.py`
3. Add tests in `tests/unit/test_validators.py`

### New Vector Store
1. Create store in `src/gweta/ingest/stores/`
2. Inherit from `BaseStore`
3. Add optional dependency in `pyproject.toml`
4. Add tests

### New MCP Tool
1. Add tool function in `src/gweta/mcp/tools.py`
2. Register with `@mcp.tool()`
3. Update documentation

## Questions?

Feel free to open an issue for questions or discussions.
