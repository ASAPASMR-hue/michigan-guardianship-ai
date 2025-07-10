# Michigan Minor Guardianship AI Assistant

An AI-powered assistant helping Michigan families navigate minor guardianship procedures with zero hallucination and maximum actionability, specifically tailored for Genesee County.

## Overview

This project implements a production-ready RAG (Retrieval-Augmented Generation) system that provides accurate, actionable guidance for Michigan minor guardianship cases. The system combines:

- **Stanford Justice Innovation's human-centered quality rubric**
- **HyPA-RAG adaptive retrieval insights**
- **Strict legal compliance with Michigan Rule of Professional Conduct 7.1**
- **Dynamic mode switching between legal facts and empathetic guidance**

## Key Features

- 🎯 **Zero Hallucination Policy**: Every legal fact has inline citations
- 🏛️ **Genesee County Specific**: Local court details, fees, and procedures
- ⚡ **Adaptive Complexity**: Simple questions get fast answers; complex scenarios get thorough analysis
- 🔒 **Privacy-First**: PII protection and data anonymization
- 📊 **Quality Assured**: Automated evaluation against human-centered rubric

## Quick Start

### Prerequisites

- Python 3.10+
- Google Cloud account with Gemini API access
- 16GB RAM (32GB recommended for local LLM inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/michigan-guardianship-ai.git
cd michigan-guardianship-ai

# Set up virtual environment
make setup-dev

# Activate virtual environment
source venv/bin/activate

# Set up environment variables
export GEMINI_API_KEY="your-api-key"
```

### Running the System

```bash
# Run the orchestrator to process questions
make run-orchestrator

# Validate configurations
make validate-configs

# Run tests
make test
```

## Project Structure

```
michigan-guardianship-ai/
├── config/                # Auto-generated YAML configurations
├── constants/             # Genesee County constants
├── data/                  # Knowledge base, rubrics, questions
├── docs/                  # Project documentation
├── patterns/              # Out-of-scope detection patterns
├── results/               # Evaluation results and reports
├── rubrics/               # Evaluation rubrics
├── scripts/               # Core implementation
├── tests/                 # Test suite
└── .github/workflows/     # CI/CD pipelines
```

## Core Components

### 1. Document Processing
- Semantic chunking with legal pattern recognition
- Metadata enrichment for precise filtering
- Incremental embedding updates

### 2. Retrieval Pipeline
- Hybrid search (70% vector + 30% lexical)
- Adaptive top-k based on query complexity
- Mandatory Genesee County filtering

### 3. Generation
- Dynamic mode switching (strict legal vs. personalized)
- Inline citation enforcement
- Out-of-scope query handling

### 4. Validation
- Hallucination detection (<5% threshold)
- Citation compliance (100% required)
- Procedural accuracy (>98% required)

## Evaluation Framework

The system is evaluated across 7 dimensions totaling 10 points:

1. **Procedural Accuracy** (2.5 pts) - Court forms, fees, deadlines
2. **Substantive Legal Accuracy** (2.0 pts) - Statutes, requirements
3. **Actionability** (2.0 pts) - Concrete next steps
4. **Mode Effectiveness** (1.5 pts) - Appropriate tone/style
5. **Strategic Caution** (0.5 pts) - Risk warnings
6. **Citation Quality** (0.5 pts) - Proper inline citations
7. **Harm Prevention** (0.5 pts) - No dangerous advice

## Genesee County Specifics

- **Filing Fee**: $175 (waiver via Form MC 20)
- **Hearings**: Thursdays only
- **Court**: 900 S. Saginaw St., Room 502, Flint, MI 48502
- **Service Deadlines**: 7 days (personal), 14 days (mail)

## Development

```bash
# Format code
make format

# Run linting
make lint

# Run full CI checks locally
make ci-local
```

## Documentation

Full documentation available in `docs/Project_Guidance_v2.1.md`

Key documents:
- Project guidance and technical specifications
- Out-of-scope handling guidelines
- Dynamic mode examples
- Genesee County specifics

## Security & Privacy

- All PII is masked or anonymized
- Query logs retained for 90 days only
- AES-256 encryption at rest
- TLS 1.3 in transit

## License

[License information to be added]

## Contributing

[Contribution guidelines to be added]

## Support

For issues or questions, please open a GitHub issue.

---

**Legal Disclaimer**: This AI assistant provides general information about Michigan minor guardianship procedures. This is NOT legal advice and does not create an attorney-client relationship. For advice about your specific situation, consult a licensed Michigan attorney.