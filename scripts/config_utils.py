"""
Configuration utilities for Michigan Guardianship AI
Handles environment variables and sensitive configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_hf_token():
    """Load Hugging Face token from environment"""
    # Load .env file if it exists
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError(
            "HF_TOKEN not found. Please set it in one of these ways:\n"
            "1. Create a .env file with HF_TOKEN=your_token\n"
            "2. Set HF_TOKEN environment variable\n"
            "3. Run: export HF_TOKEN='your_token_here'\n"
            "4. See SETUP.md for detailed instructions"
        )
    return hf_token

def get_model_config():
    """Get model configuration based on environment"""
    use_small_model = os.getenv('USE_SMALL_MODEL', 'false').lower() == 'true'
    
    if use_small_model:
        return {
            'primary_model': 'all-MiniLM-L6-v2',
            'fallback_model': 'paraphrase-MiniLM-L6-v2'
        }
    else:
        return {
            'primary_model': 'BAAI/bge-m3',
            'fallback_model': 'intfloat/multilingual-e5-large'
        }
