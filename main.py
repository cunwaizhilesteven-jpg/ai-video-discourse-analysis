"""
AI-Generated Video Comment Analysis - Main Entry Point

Analyzes user acceptance and engagement with AI-generated videos using
sentiment analysis and linguistic complexity on YouTube comments.

Usage:
    python main.py <channel_url> [--max-videos N] [--skip-collect] [--skip-analyze]

Example:
    python main.py https://www.youtube.com/@ReallyNotAi --max-videos 5
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.collect import YouTubeCommentCollector
from scripts.processor import CommentProcessor
from scripts.analyzer import CommentAnalyzer
from scripts.visualizer import CommentVisualizer


def main(channel_url: str,
         max_videos: int = None,
         skip_collect: bool = False,
         skip_analyze: bool = False):
    """
    Run the complete analysis pipeline.

    Args:
        channel_url: YouTube channel URL
        max_videos: Maximum number of videos to process (None for all)
        skip_collect: Skip data collection step (use existing data)
        skip_analyze: Skip analysis step (use existing analyzed data)
    """
    print("=" * 60)
    print("AI-Generated Video Comment Analysis")
    print("=" * 60)
    print(f"Channel: {channel_url}")
    print(f"Max videos: {max_videos if max_videos else 'All'}")
    print("=" * 60)

    # Step 1: Collect comments
    if not skip_collect:
        print("\n[Step 1/4] Collecting YouTube comments...")
        print("-" * 40)
        collector = YouTubeCommentCollector(
            channel_url=channel_url,
            output_dir="data/raw_json",
            progress_dir="data/progress",
            log_dir="logs"
        )
        collection_result = collector.run(max_videos=max_videos)

        if collection_result['success'] == 0 and collection_result['skipped'] == 0:
            print("No comments collected. Exiting.")
            return
    else:
        print("\n[Step 1/4] Skipping data collection (using existing data)")

    # Step 2: Process data
    print("\n[Step 2/4] Processing and cleaning data...")
    print("-" * 40)
    processor = CommentProcessor(
        raw_dir="data/raw_json",
        output_path="data/merged_data.csv",
        log_dir="logs"
    )
    df = processor.run()

    if df.empty:
        print("No data to analyze. Exiting.")
        return

    # Step 3: Analyze sentiment and complexity
    if not skip_analyze:
        print("\n[Step 3/4] Analyzing sentiment and linguistic complexity...")
        print("-" * 40)
        analyzer = CommentAnalyzer(
            input_path="data/merged_data.csv",
            output_path="data/merged_data.csv",
            log_dir="logs",
            batch_size=32
        )
        df = analyzer.run()
    else:
        print("\n[Step 3/4] Skipping analysis (using existing analyzed data)")

    # Step 4: Generate visualizations
    print("\n[Step 4/4] Generating visualizations...")
    print("-" * 40)
    visualizer = CommentVisualizer(
        input_path="data/merged_data.csv",
        output_dir="output",
        log_dir="logs"
    )
    visualizer.run()

    # Final summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - Data: data/merged_data.csv")
    print("  - Charts (PNG): output/png/")
    print("  - Charts (HTML): output/html/")
    print("  - Logs: logs/")
    print("\nGenerated visualizations:")
    print("  1. Sentiment Distribution (pie chart)")
    print("  2. Engagement vs Acceptance (scatter plot)")
    print("  3. Negative Comments Word Cloud")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze YouTube comments on AI-generated videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py https://www.youtube.com/@ReallyNotAi
  python main.py https://www.youtube.com/@ReallyNotAi --max-videos 5
  python main.py https://www.youtube.com/@ReallyNotAi --skip-collect
        """
    )

    parser.add_argument(
        "channel_url",
        help="YouTube channel URL (e.g., https://www.youtube.com/@ChannelName)"
    )

    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of videos to process (default: all)"
    )

    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip data collection step (use existing raw data)"
    )

    parser.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip analysis step (use existing analyzed data)"
    )

    args = parser.parse_args()

    main(
        channel_url=args.channel_url,
        max_videos=args.max_videos,
        skip_collect=args.skip_collect,
        skip_analyze=args.skip_analyze
    )
