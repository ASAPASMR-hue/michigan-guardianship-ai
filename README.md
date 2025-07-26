# Michigan Guardianship AI

A production-ready RAG (Retrieval-Augmented Generation) system for Genesee County's minor guardianship procedures.

## Overview

This system provides accurate, accessible information about minor guardianship procedures, requirements, and forms specific to Genesee County, Michigan. It features:

- Zero hallucination policy with citation verification
- Adaptive retrieval with keyword boosting
- Semantic similarity validation
- Support for ICWA (Indian Child Welfare Act) requirements
- Integration with multiple LLMs for testing

## Setup

### Prerequisites

- Python 3.8+
- ChromaDB
- HuggingFace account (for some embedding models)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ASAPASMR-hue/michigan-guardianship-ai.git
cd michigan-guardianship-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your tokens:
# - HUGGING_FACE_HUB_TOKEN: Get from https://huggingface.co/settings/tokens
# - GEMINI_API_KEY: Get from https://makersuite.google.com/app/apikey
```

4. Export the environment variables:
```bash
export HUGGING_FACE_HUB_TOKEN='your_token_here'
export GEMINI_API_KEY='your_api_key_here'
```

## Usage

### Running the Full Pipeline

```bash
# Run Phase 1 pipeline (embedding, retrieval, validation)
make phase1

# Run with small models for testing
make test-phase1

# Clean and rebuild everything
make clean-all && make phase1
```

### Testing

```bash
# Run integration tests
python integration_tests/run_golden_qa.py

# Run end-to-end test with Google Gemini
python test_gemini_e2e.py
```

## Key Features

### 1. Document Processing
- Semantic chunking with overlap
- Metadata extraction
- BAAI/bge-m3 embeddings

### 2. Adaptive Retrieval
- Query complexity classification
- Dynamic parameter adjustment
- Hybrid search (vector + BM25)
- Keyword boosting for critical terms

### 3. Response Validation
- Hallucination detection
- Citation verification
- Procedural accuracy checks

### 4. Genesee County Specifics
- Filing fee: $175 (fee waiver via Form MC 20)
- Court location: 900 S. Saginaw Street, Flint, MI 48502
- Hearings: Thursdays at 9:00 AM

## Project Structure

```
michigan-guardianship-ai/
├── scripts/              # Core pipeline scripts
│   ├── embed_kb.py      # Document embedding
│   ├── retrieval_setup.py # Retrieval system
│   └── validator_setup.py # Response validation
├── integration_tests/    # Test suites
├── config/              # Configuration files
├── kb_files/            # Knowledge base documents
└── docs/                # Documentation
```

## Recent Improvements

- Implemented semantic similarity checking (0.8 threshold) for test validation
- Added keyword boosting (1.3-1.5x) for critical terms
- Enhanced query expansion with legal term mappings
- Improved test pass rate to 70% with mock tests and 60% with real LLM

## License

[License information here]

## Contact

[Contact information here]