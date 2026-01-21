# CLAUDE.md - ai-video-discourse-analysis

> **Documentation Version**: 1.0
> **Last Updated**: 2026-01-21
> **Project**: ai-video-discourse-analysis
> **Description**: Analyzing user acceptance and engagement with AI-generated videos using sentiment analysis and linguistic complexity on YouTube comments.

This file provides essential guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL RULES - READ FIRST

> **RULE ADHERENCE SYSTEM ACTIVE**
> **Claude Code must explicitly acknowledge these rules at task start**

### RULE ACKNOWLEDGMENT REQUIRED
> **Before starting ANY task, Claude Code must respond with:**
> "CRITICAL RULES ACKNOWLEDGED - I will follow all prohibitions and requirements listed in CLAUDE.md"

### ABSOLUTE PROHIBITIONS
- **NEVER** write output files directly to root directory - use designated output folders
- **NEVER** create documentation files (.md) unless explicitly requested by user
- **NEVER** use git commands with -i flag (interactive mode not supported)
- **NEVER** use `find`, `grep`, `cat`, `head`, `tail`, `ls` commands - use Read, Grep, Glob tools instead
- **NEVER** create duplicate files (manager_v2.py, enhanced_xyz.py, utils_new.js) - ALWAYS extend existing files
- **NEVER** create multiple implementations of same concept - single source of truth
- **NEVER** use naming like enhanced_, improved_, new_, v2_ - extend original files instead

### MANDATORY REQUIREMENTS
- **COMMIT** after every completed task/phase - no exceptions
- **GITHUB BACKUP** - Push to GitHub after every commit: `git push origin main`
- **USE TASK AGENTS** for all long-running operations (>30 seconds)
- **TODOWRITE** for complex tasks (3+ steps)
- **READ FILES FIRST** before editing
- **DEBT PREVENTION** - Before creating new files, check for existing similar functionality to extend
- **SINGLE SOURCE OF TRUTH** - One authoritative implementation per feature/concept

## PROJECT OVERVIEW

### Project Purpose
This project analyzes YouTube comments on AI-generated videos to understand:
- User sentiment towards AI-generated content
- Linguistic complexity patterns in user responses
- Engagement metrics and acceptance indicators

### Technology Stack
- **Language**: Python 3.10+
- **Data Collection**: yt-dlp, youtube-comment-downloader
- **ML/NLP**: transformers (DistilBERT), textstat
- **Data Processing**: pandas
- **Visualization**: matplotlib, seaborn, plotly, wordcloud

### Development Status
- **Setup**: Complete
- **Data Collection**: Complete (collect.py)
- **Data Processing**: Complete (processor.py)
- **Sentiment Analysis**: Complete (analyzer.py)
- **Visualization**: Complete (visualizer.py)

## PROJECT STRUCTURE

```
ai-video-discourse-analysis/
├── main.py                # Main entry point - orchestrates all modules
├── requirements.txt       # Python dependencies
├── scripts/               # Core modules
│   ├── collect.py         # YouTube comment collection with checkpoint support
│   ├── processor.py       # JSON to CSV conversion and cleaning
│   ├── analyzer.py        # Sentiment analysis + linguistic complexity
│   └── visualizer.py      # Chart generation (PNG + HTML)
├── data/
│   ├── raw_json/          # Individual JSON files per video
│   ├── progress/          # Checkpoint files (completed.txt, failed.txt)
│   └── merged_data.csv    # Final cleaned and analyzed dataset
├── output/
│   ├── png/               # Static chart images
│   └── html/              # Interactive charts
└── logs/                  # Execution logs
```

## COMMON COMMANDS

```bash
# Activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py https://www.youtube.com/@ReallyNotAi

# Run with limited videos (for testing)
python main.py https://www.youtube.com/@ReallyNotAi --max-videos 5

# Skip collection (use existing data)
python main.py https://www.youtube.com/@ReallyNotAi --skip-collect

# Run individual modules
python scripts/collect.py https://www.youtube.com/@ReallyNotAi
python scripts/processor.py
python scripts/analyzer.py
python scripts/visualizer.py
```

## MODULE DETAILS

### collect.py - YouTubeCommentCollector
- `get_video_ids()` - Extract video IDs from channel
- `load_progress()` / `save_progress()` - Checkpoint support
- `download_comments()` - Download with retry logic
- `run()` - Main execution with resume support

### processor.py - CommentProcessor
- `load_json_files()` - Read all JSON files
- `extract_fields()` - Extract: text, votes, replies, time, video_id
- `remove_duplicates()` - Dedupe by comment ID and text
- `remove_emoji_only()` - Filter emoji-only comments
- `run()` - Output: merged_data.csv

### analyzer.py - CommentAnalyzer
- `load_model()` - Load DistilBERT (GPU if available)
- `analyze_sentiment_batch()` - Batch sentiment analysis
- `calculate_complexity()` - Lexical density, Flesch score
- `run()` - Add: sentiment_label, sentiment_score, lexical_density

### visualizer.py - CommentVisualizer
- `plot_sentiment_pie()` - Sentiment distribution
- `plot_engagement_scatter()` - Lexical density vs sentiment
- `plot_negative_wordcloud()` - Negative comment keywords
- `run()` - Generate PNG + HTML outputs

## TECHNICAL DEBT PREVENTION

### Before Creating ANY New File:
1. **Search First** - Use Grep/Glob to find existing implementations
2. **Analyze Existing** - Read and understand current patterns
3. **Extend Existing** - Prefer extending over creating new
4. **Follow Patterns** - Use established project patterns

---

**Prevention is better than consolidation - build clean from the start.**
