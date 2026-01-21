"""
Data Collection Module - YouTube Comment Collector

Collects comments from all videos of a YouTube channel using yt-dlp and youtube-comment-downloader.
Supports checkpoint/resume functionality for robust data collection.
"""

import os
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime


class YouTubeCommentCollector:
    """Collects YouTube comments from a channel with checkpoint support."""

    def __init__(self, channel_url: str, output_dir: str = "data/raw_json",
                 progress_dir: str = "data/progress", log_dir: str = "logs"):
        """
        Initialize the collector.

        Args:
            channel_url: YouTube channel URL (e.g., https://www.youtube.com/@ChannelName)
            output_dir: Directory to save comment JSON files
            progress_dir: Directory to save progress files for checkpoint/resume
            log_dir: Directory for log files
        """
        self.channel_url = channel_url
        self.output_dir = Path(output_dir)
        self.progress_dir = Path(progress_dir)
        self.log_dir = Path(log_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Progress files
        self.completed_file = self.progress_dir / "completed.txt"
        self.failed_file = self.progress_dir / "failed.txt"

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to file and console."""
        log_file = self.log_dir / f"collect_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_video_ids(self) -> list:
        """
        Get all video IDs from the channel using yt-dlp.

        Returns:
            List of video IDs
        """
        self.logger.info(f"Fetching video IDs from channel: {self.channel_url}")

        try:
            # Use yt-dlp to get video IDs
            # Use channel URL directly (yt-dlp handles it)
            cmd = [
                "yt-dlp",
                "--get-id",
                "--flat-playlist",
                self.channel_url
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                self.logger.error(f"yt-dlp error: {result.stderr}")
                return []

            video_ids = [vid.strip() for vid in result.stdout.strip().split('\n') if vid.strip()]
            self.logger.info(f"Found {len(video_ids)} videos")
            return video_ids

        except subprocess.TimeoutExpired:
            self.logger.error("Timeout while fetching video IDs")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching video IDs: {e}")
            return []

    def load_progress(self) -> set:
        """
        Load completed video IDs from checkpoint file.

        Returns:
            Set of completed video IDs
        """
        completed = set()
        if self.completed_file.exists():
            with open(self.completed_file, 'r', encoding='utf-8') as f:
                completed = set(line.strip() for line in f if line.strip())
            self.logger.info(f"Loaded {len(completed)} completed video IDs from checkpoint")
        return completed

    def save_progress(self, video_id: str, success: bool = True):
        """
        Save progress for a video ID.

        Args:
            video_id: The video ID to save
            success: Whether the download was successful
        """
        target_file = self.completed_file if success else self.failed_file
        with open(target_file, 'a', encoding='utf-8') as f:
            f.write(f"{video_id}\n")

    def download_comments(self, video_id: str, max_retries: int = 3) -> bool:
        """
        Download comments for a single video with retry logic.

        Args:
            video_id: YouTube video ID
            max_retries: Maximum number of retry attempts

        Returns:
            True if successful, False otherwise
        """
        output_file = self.output_dir / f"{video_id}.json"

        # Skip if already downloaded
        if output_file.exists():
            self.logger.info(f"Skipping {video_id} - already downloaded")
            return True

        retry_delays = [1, 3, 5]  # Increasing delays between retries

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading comments for {video_id} (attempt {attempt + 1}/{max_retries})")

                cmd = [
                    "youtube-comment-downloader",
                    "--youtubeid", video_id,
                    "--output", str(output_file)
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180  # 3 minute timeout per video
                )

                if result.returncode == 0 and output_file.exists():
                    self.logger.info(f"Successfully downloaded comments for {video_id}")
                    return True
                else:
                    self.logger.warning(f"Failed to download {video_id}: {result.stderr}")

            except subprocess.TimeoutExpired:
                self.logger.warning(f"Timeout downloading comments for {video_id}")
            except Exception as e:
                self.logger.warning(f"Error downloading {video_id}: {e}")

            # Wait before retry (if not last attempt)
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                self.logger.info(f"Waiting {delay}s before retry...")
                time.sleep(delay)

        self.logger.error(f"Failed to download comments for {video_id} after {max_retries} attempts")
        return False

    def run(self, max_videos: int = None) -> dict:
        """
        Main execution logic with checkpoint/resume support.

        Args:
            max_videos: Maximum number of videos to process (None for all)

        Returns:
            Summary dict with counts of successful/failed downloads
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting YouTube Comment Collection")
        self.logger.info(f"Channel: {self.channel_url}")
        self.logger.info("=" * 50)

        # Get all video IDs
        video_ids = self.get_video_ids()
        if not video_ids:
            self.logger.error("No videos found. Exiting.")
            return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

        # Load checkpoint
        completed = self.load_progress()

        # Filter out already completed videos
        pending_ids = [vid for vid in video_ids if vid not in completed]

        # Apply max_videos limit
        if max_videos:
            pending_ids = pending_ids[:max_videos]

        self.logger.info(f"Total videos: {len(video_ids)}")
        self.logger.info(f"Already completed: {len(completed)}")
        self.logger.info(f"To process: {len(pending_ids)}")

        # Process videos
        success_count = 0
        failed_count = 0

        for i, video_id in enumerate(pending_ids, 1):
            self.logger.info(f"Processing [{i}/{len(pending_ids)}]: {video_id}")

            success = self.download_comments(video_id)
            self.save_progress(video_id, success)

            if success:
                success_count += 1
            else:
                failed_count += 1

            # Small delay between videos to avoid rate limiting
            if i < len(pending_ids):
                time.sleep(0.5)

        # Summary
        summary = {
            "total": len(video_ids),
            "success": success_count,
            "failed": failed_count,
            "skipped": len(completed)
        }

        self.logger.info("=" * 50)
        self.logger.info("Collection Complete!")
        self.logger.info(f"Total videos: {summary['total']}")
        self.logger.info(f"Successfully downloaded: {summary['success']}")
        self.logger.info(f"Failed: {summary['failed']}")
        self.logger.info(f"Previously completed: {summary['skipped']}")
        self.logger.info("=" * 50)

        return summary


def main():
    """Entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Download YouTube comments from a channel")
    parser.add_argument("channel_url", help="YouTube channel URL")
    parser.add_argument("--max-videos", type=int, default=None, help="Maximum videos to process")
    parser.add_argument("--output-dir", default="data/raw_json", help="Output directory for JSON files")

    args = parser.parse_args()

    collector = YouTubeCommentCollector(
        channel_url=args.channel_url,
        output_dir=args.output_dir
    )
    collector.run(max_videos=args.max_videos)


if __name__ == "__main__":
    main()
