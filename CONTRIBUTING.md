# Contributing to Michigan Guardianship AI

## Security Guidelines

### API Keys and Tokens

**NEVER** commit API keys, tokens, or other sensitive information to the repository.

1. Always use environment variables for sensitive data
2. Add any files containing secrets to `.gitignore`
3. Use the `config_utils.py` module for loading tokens
4. If you accidentally commit a token:
   - Immediately revoke the token
   - Remove it from the repository history
   - Generate a new token

### Before Submitting a Pull Request

1. Run `git diff` to review all changes
2. Search for any hardcoded tokens: `grep -r "hf_" . --exclude-dir=.git`
3. Ensure all tests pass with `make test-phase1`
4. Update documentation if needed

## Development Setup

See [SETUP.md](SETUP.md) for detailed setup instructions.

## Code Style

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to all functions
- Keep functions focused and small

## Testing

- Write tests for new functionality
- Ensure existing tests pass
- Test with both full and small models

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Commit with clear messages: `git commit -m "Add: feature description"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Create a pull request with a clear description

## Questions?

Open an issue for any questions or concerns.