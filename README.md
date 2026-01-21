# ai-video-discourse-analysis

Analyzing user acceptance and engagement with AI-generated videos using sentiment analysis and linguistic complexity on YouTube comments.

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full analysis pipeline
python main.py https://www.youtube.com/@ReallyNotAi

# Or with limited videos for testing
python main.py https://www.youtube.com/@ReallyNotAi --max-videos 5
```

## Project Structure

```
ai-video-discourse-analysis/
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
├── scripts/               # Core modules
│   ├── collect.py         # YouTube comment collection (yt-dlp + youtube-comment-downloader)
│   ├── processor.py       # Data cleaning and consolidation
│   ├── analyzer.py        # Sentiment analysis (DistilBERT) + linguistic complexity
│   └── visualizer.py      # Visualization generation (PNG + HTML)
├── data/
│   ├── raw_json/          # Individual JSON files per video
│   ├── progress/          # Checkpoint files for resume support
│   └── merged_data.csv    # Final cleaned and analyzed dataset
├── output/
│   ├── png/               # Static chart images
│   └── html/              # Interactive charts
└── logs/                  # Execution logs
```

## Modules

### 1. Data Collection (`scripts/collect.py`)
- Extracts all video IDs from a YouTube channel using `yt-dlp`
- Downloads comments for each video using `youtube-comment-downloader`
- Supports checkpoint/resume for interrupted downloads
- Automatic retry with exponential backoff

```bash
# Standalone usage
python scripts/collect.py https://www.youtube.com/@ReallyNotAi --max-videos 10
```

### 2. Data Processing (`scripts/processor.py`)
- Consolidates JSON files into a single CSV
- Extracts fields: text, votes, replies, time, video_id
- Removes duplicates and emoji-only comments

```bash
# Standalone usage
python scripts/processor.py --raw-dir data/raw_json --output data/merged_data.csv
```

### 3. Analysis (`scripts/analyzer.py`)
- Sentiment analysis using `distilbert-base-uncased-finetuned-sst-2-english`
- Linguistic complexity using `textstat` (lexical density, Flesch reading ease)
- Batch processing with GPU support

```bash
# Standalone usage
python scripts/analyzer.py --input data/merged_data.csv
```

### 4. Visualization (`scripts/visualizer.py`)
- Chart 1: Sentiment distribution pie chart
- Chart 2: Engagement (lexical density) vs Acceptance (sentiment) scatter plot
- Chart 3: Negative comments word cloud
- Outputs both PNG and interactive HTML formats

```bash
# Standalone usage
python scripts/visualizer.py --input data/merged_data.csv --output-dir output
```

## Output

After running the full pipeline, you will find:

| File | Description |
|------|-------------|
| `data/merged_data.csv` | All comments with sentiment labels and complexity scores |
| `output/png/sentiment_distribution.png` | Sentiment pie chart |
| `output/png/engagement_scatter.png` | Engagement vs acceptance scatter plot |
| `output/png/negative_wordcloud.png` | Word cloud of negative comments |
| `output/html/sentiment_distribution.html` | Interactive pie chart |
| `output/html/engagement_scatter.html` | Interactive scatter plot |

## Command Line Options

```bash
python main.py <channel_url> [options]

Options:
  --max-videos N    Limit number of videos to process (default: all)
  --skip-collect    Skip data collection (use existing raw data)
  --skip-analyze    Skip analysis (use existing analyzed data)
```

## Dependencies

- `yt-dlp` - YouTube video ID extraction
- `youtube-comment-downloader` - Comment downloading
- `pandas` - Data processing
- `transformers` + `torch` - DistilBERT sentiment analysis
- `textstat` - Linguistic complexity metrics
- `matplotlib` + `seaborn` - Static visualizations
- `plotly` - Interactive visualizations
- `wordcloud` - Word cloud generation

## License

[Add license information]
