# ai-video-discourse-analysis

Analyzing user acceptance and engagement with AI-generated videos using sentiment analysis and linguistic complexity on YouTube comments.

## Quick Start

1. **Read CLAUDE.md first** - Contains essential rules for Claude Code
2. Follow the pre-task compliance checklist before starting any work
3. Use proper module structure under `src/main/python/`
4. Commit after every completed task

## Project Structure

```
ai-video-discourse-analysis/
├── CLAUDE.md              # Essential rules for Claude Code
├── README.md              # Project documentation
├── .gitignore             # Git ignore patterns
├── src/                   # Source code
│   ├── main/
│   │   ├── python/        # Python source code
│   │   │   ├── core/      # Core sentiment analysis algorithms
│   │   │   ├── utils/     # Data processing utilities
│   │   │   ├── models/    # Model definitions
│   │   │   ├── services/  # Analysis services and pipelines
│   │   │   ├── api/       # API endpoints
│   │   │   ├── training/  # Training scripts
│   │   │   ├── inference/ # Inference code
│   │   │   └── evaluation/# Model evaluation
│   │   └── resources/     # Configuration and assets
│   └── test/              # Test code
├── data/                  # Dataset management
│   ├── raw/               # Raw YouTube comment data
│   ├── processed/         # Cleaned and transformed data
│   ├── external/          # External data sources
│   └── temp/              # Temporary files
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/       # Data exploration
│   ├── experiments/       # ML experiments
│   └── reports/           # Analysis reports
├── models/                # ML model artifacts
│   ├── trained/           # Trained models
│   ├── checkpoints/       # Model checkpoints
│   └── metadata/          # Model metadata
├── experiments/           # Experiment tracking
├── docs/                  # Documentation
├── output/                # Generated output files
└── logs/                  # Log files
```

## Development Guidelines

- **Always search first** before creating new files
- **Extend existing** functionality rather than duplicating
- **Use Task agents** for operations >30 seconds
- **Single source of truth** for all functionality
- **Commit after each feature** with descriptive messages

## Key Features

- YouTube comment extraction and preprocessing
- Sentiment analysis on AI-generated video comments
- Linguistic complexity measurement
- User engagement metrics analysis
- Comparative analysis between AI and human-created content

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when requirements.txt is created)
pip install -r requirements.txt
```

## License

[Add license information]
