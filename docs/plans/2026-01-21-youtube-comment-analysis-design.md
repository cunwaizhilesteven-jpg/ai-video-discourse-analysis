# YouTube Comment Analysis - Design Document

> **Date**: 2026-01-21
> **Status**: Implemented
> **Project**: ai-video-discourse-analysis

## Overview

Analyze user acceptance and engagement with AI-generated videos through YouTube comment analysis using sentiment analysis and linguistic complexity metrics.

## Design Decisions

### Data Collection
- **Tool**: youtube-comment-downloader (no API key required)
- **Error Handling**: Checkpoint/resume with 3 retries, exponential backoff (1s, 3s, 5s)
- **Progress Files**: `data/progress/completed.txt`, `data/progress/failed.txt`

### Sentiment Analysis
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Output**: POSITIVE/NEGATIVE label + confidence score (0-1)
- **Batch Processing**: batch_size=32 with GPU support

### Linguistic Complexity
- **Tool**: textstat
- **Metrics**: Lexical density, Flesch reading ease, word count

### Visualization
- **Format**: PNG (static) + HTML (interactive)
- **Charts**:
  1. Sentiment distribution pie chart
  2. Engagement (lexical density) vs Acceptance (sentiment) scatter plot
  3. Negative comments word cloud

## Project Structure

```
ai-video-discourse-analysis/
├── main.py                # Main entry point
├── requirements.txt       # Dependencies
├── scripts/
│   ├── collect.py         # YouTubeCommentCollector class
│   ├── processor.py       # CommentProcessor class
│   ├── analyzer.py        # CommentAnalyzer class
│   └── visualizer.py      # CommentVisualizer class
├── data/
│   ├── raw_json/          # Per-video JSON files
│   ├── progress/          # Checkpoint files
│   └── merged_data.csv    # Final dataset
├── output/
│   ├── png/               # Static charts
│   └── html/              # Interactive charts
└── logs/                  # Execution logs
```

## Usage

```bash
# Full pipeline
python main.py https://www.youtube.com/@ReallyNotAi

# With video limit (testing)
python main.py https://www.youtube.com/@ReallyNotAi --max-videos 5

# Skip collection (use existing data)
python main.py https://www.youtube.com/@ReallyNotAi --skip-collect
```

## Data Flow

```
YouTube Channel URL
       │
       ▼
┌──────────────┐
│  collect.py  │ → data/raw_json/*.json
└──────────────┘
       │
       ▼
┌──────────────┐
│ processor.py │ → data/merged_data.csv (text, votes, replies, time, video_id)
└──────────────┘
       │
       ▼
┌──────────────┐
│  analyzer.py │ → data/merged_data.csv (+sentiment_label, sentiment_score, lexical_density)
└──────────────┘
       │
       ▼
┌──────────────┐
│visualizer.py │ → output/png/*.png, output/html/*.html
└──────────────┘
```

## Output Fields (merged_data.csv)

| Field | Source | Description |
|-------|--------|-------------|
| video_id | collect | YouTube video ID |
| text | collect | Comment text |
| votes | collect | Like count |
| replies | collect | Reply count |
| time | collect | Post time |
| author | collect | Commenter name |
| sentiment_label | analyzer | POSITIVE/NEGATIVE |
| sentiment_score | analyzer | Confidence (0-1) |
| lexical_density | analyzer | Unique words / total words |
| flesch_reading_ease | analyzer | Readability score |
| word_count | analyzer | Total words |
