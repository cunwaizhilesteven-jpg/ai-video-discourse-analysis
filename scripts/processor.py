"""
Data Processing Module - Comment Processor

Consolidates JSON comment files into a single CSV dataset with cleaning operations.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd


class CommentProcessor:
    """Processes raw JSON comment files into a cleaned CSV dataset."""

    def __init__(self, raw_dir: str = "data/raw_json",
                 output_path: str = "data/merged_data.csv",
                 metadata_path: str = "data/video_metadata.csv",
                 log_dir: str = "logs"):
        """
        Initialize the processor.

        Args:
            raw_dir: Directory containing raw JSON files
            output_path: Path for the output CSV file
            log_dir: Directory for log files
        """
        self.raw_dir = Path(raw_dir)
        self.output_path = Path(output_path)
        self.metadata_path = Path(metadata_path)
        self.log_dir = Path(log_dir)

        # Create directories
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to file and console."""
        log_file = self.log_dir / f"processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_json_files(self) -> list:
        """
        Load all JSON files from the raw directory.

        Returns:
            List of tuples (video_id, comments_list)
        """
        self.logger.info(f"Loading JSON files from: {self.raw_dir}")

        all_data = []
        json_files = list(self.raw_dir.glob("*.json"))

        if not json_files:
            self.logger.warning("No JSON files found!")
            return all_data

        self.logger.info(f"Found {len(json_files)} JSON files")

        for json_file in json_files:
            video_id = json_file.stem  # filename without extension
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    # youtube-comment-downloader outputs one JSON object per line
                    comments = []
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                comment = json.loads(line)
                                comments.append(comment)
                            except json.JSONDecodeError:
                                continue

                    if comments:
                        all_data.append((video_id, comments))
                        self.logger.info(f"Loaded {len(comments)} comments from {video_id}")

            except Exception as e:
                self.logger.error(f"Error loading {json_file}: {e}")

        return all_data

    def extract_fields(self, all_data: list) -> pd.DataFrame:
        """
        Extract core fields from comments into a DataFrame.

        Args:
            all_data: List of tuples (video_id, comments_list)

        Returns:
            DataFrame with extracted fields
        """
        self.logger.info("Extracting core fields from comments")

        records = []
        for video_id, comments in all_data:
            for comment in comments:
                record = {
                    'video_id': video_id,
                    'text': comment.get('text', ''),
                    'votes': comment.get('votes', 0),
                    'replies': comment.get('replies', 0),
                    'time': comment.get('time', ''),
                    'author': comment.get('author', ''),
                    'cid': comment.get('cid', '')  # comment ID for deduplication
                }
                records.append(record)

        df = pd.DataFrame(records)
        self.logger.info(f"Extracted {len(df)} comments total")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate comments based on comment ID and text.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with duplicates removed
        """
        original_count = len(df)

        # Remove duplicates by comment ID first (if available)
        if 'cid' in df.columns:
            df = df.drop_duplicates(subset=['cid'], keep='first')

        # Also remove duplicates by text content (same text, different IDs)
        df = df.drop_duplicates(subset=['text'], keep='first')

        removed = original_count - len(df)
        self.logger.info(f"Removed {removed} duplicate comments ({len(df)} remaining)")
        return df

    def remove_emoji_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove comments that are completely empty or whitespace-only.
        Emoji-only comments and low-frequency words are preserved.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with blank-only comments removed
        """
        original_count = len(df)

        df = df[df['text'].apply(lambda t: isinstance(t, str) and t.strip() != '')]

        removed = original_count - len(df)
        self.logger.info(f"Removed {removed} blank comments ({len(df)} remaining)")
        return df

    def merge_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge video metadata (title, hashtags, ai_explicit) into the comment DataFrame.

        Args:
            df: Input DataFrame with video_id column

        Returns:
            DataFrame with video_title, title_hashtags, ai_explicit columns added
        """
        if not self.metadata_path.exists():
            self.logger.warning(f"Video metadata file not found: {self.metadata_path}")
            df['video_title'] = ''
            df['title_hashtags'] = ''
            df['ai_explicit'] = False
            return df

        metadata = pd.read_csv(self.metadata_path, encoding='utf-8')
        self.logger.info(f"Loaded metadata for {len(metadata)} videos")

        before_cols = set(df.columns)
        df = df.merge(metadata[['video_id', 'video_title', 'title_hashtags', 'ai_explicit']],
                       on='video_id', how='left')

        df['video_title'] = df['video_title'].fillna('')
        df['title_hashtags'] = df['title_hashtags'].fillna('')
        df['ai_explicit'] = df['ai_explicit'].fillna(False).astype(bool)

        matched = df['video_title'].ne('').sum()
        self.logger.info(f"Matched metadata for {matched}/{len(df)} comments")

        return df

    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean comment text (normalize whitespace, etc.).

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned text
        """
        self.logger.info("Cleaning comment text")

        # Normalize whitespace
        df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

        # Remove empty comments
        df = df[df['text'].str.len() > 0]

        return df

    def run(self) -> pd.DataFrame:
        """
        Main execution logic.

        Returns:
            Cleaned DataFrame (also saved to CSV)
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting Comment Processing")
        self.logger.info("=" * 50)

        # Step 1: Load JSON files
        all_data = self.load_json_files()
        if not all_data:
            self.logger.error("No data to process. Exiting.")
            return pd.DataFrame()

        # Step 2: Extract fields
        df = self.extract_fields(all_data)

        # Step 3: Merge video metadata
        df = self.merge_metadata(df)

        # Step 4: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 5: Remove emoji-only comments
        df = self.remove_emoji_only(df)

        # Step 6: Clean text
        df = self.clean_text(df)

        # Step 7: Save to CSV
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        self.logger.info(f"Saved {len(df)} comments to {self.output_path}")

        # Summary
        self.logger.info("=" * 50)
        self.logger.info("Processing Complete!")
        self.logger.info(f"Total comments: {len(df)}")
        self.logger.info(f"Unique videos: {df['video_id'].nunique()}")
        self.logger.info(f"Output file: {self.output_path}")
        self.logger.info("=" * 50)

        return df


def main():
    """Entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Process raw JSON comments into CSV")
    parser.add_argument("--raw-dir", default="data/raw_json", help="Directory with raw JSON files")
    parser.add_argument("--output", default="data/merged_data.csv", help="Output CSV path")

    args = parser.parse_args()

    processor = CommentProcessor(
        raw_dir=args.raw_dir,
        output_path=args.output
    )
    processor.run()


if __name__ == "__main__":
    main()
