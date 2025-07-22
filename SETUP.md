# Michigan Guardianship AI - Setup Instructions

## Prerequisites

- Python 3.8 or higher
- Git
- A Hugging Face account (free)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ASAPASMR-hue/michigan-guardianship-ai.git
cd michigan-guardianship-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Hugging Face Token Configuration

This project requires a Hugging Face token to download embedding models.

### Getting a Token

1. Create an account at [https://huggingface.co](https://huggingface.co)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Click "New token"
4. Give it a descriptive name (e.g., "michigan-guardianship-ai")
5. Select "Read" permissions
6. Copy the token (it starts with `hf_`)

### Configuration Methods

#### Option 1: Environment File (Recommended for Development)

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your token:
```
HF_TOKEN=hf_your_actual_token_here
```

#### Option 2: Environment Variable

```bash
export HF_TOKEN='hf_your_actual_token_here'
```

#### Option 3: For Production

Use your deployment platform's secret management system:
- **GitHub Actions**: Add as repository secret
- **Heroku**: Add as config var
- **AWS**: Use Secrets Manager
- **Docker**: Use Docker secrets

### Verify Configuration

Test that your token is properly configured:

```bash
python -c "from scripts.config_utils import load_hf_token; print('✓ Token loaded successfully')"
```

## Running the Application

### Phase 1: Basic Setup

```bash
make phase1
```

### Testing with Small Models

For faster testing during development:

```bash
make test-phase1
```

This uses smaller models that download quickly but may have reduced accuracy.

## Troubleshooting

### Token Not Found Error

If you see `ValueError: HF_TOKEN not found`, ensure:
1. The `.env` file exists in the project root
2. The token is correctly formatted (starts with `hf_`)
3. No extra spaces or quotes in the `.env` file

### Model Download Issues

If model downloads fail:
1. Check your internet connection
2. Verify your token has read permissions
3. Try using the fallback models by setting `USE_SMALL_MODEL=true`

### Out of Memory Errors

If you encounter memory issues:
1. Use smaller models: `export USE_SMALL_MODEL=true`
2. Reduce batch size in the configuration files
3. Ensure you have at least 8GB of RAM available

## Security Best Practices

1. **Never commit tokens**: The `.env` file is gitignored
2. **Use read-only tokens**: Unless you need to push models
3. **Rotate tokens regularly**: Especially if exposed
4. **Different tokens for different environments**: Use separate tokens for dev/staging/prod

## Additional Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Project Documentation](docs/Project_Guidance_v2.1.md)
- [Contributing Guidelines](CONTRIBUTING.md)