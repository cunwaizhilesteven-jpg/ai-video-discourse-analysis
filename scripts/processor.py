"""
Data Processing Module - Comment Processor

Consolidates JSON comment files into a single CSV dataset with cleaning operations.
"""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd


class CommentProcessor:
    """Processes raw JSON comment files into a cleaned CSV dataset."""

    def __init__(self, raw_dir: str = "data/raw_json",
                 output_path: str = "data/merged_data.csv",
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
        Remove comments that contain only emojis/emoticons.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with emoji-only comments removed
        """
        original_count = len(df)

        # Pattern to match emoji-only strings
        # Remove emojis and whitespace, check if anything remains
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FA6F"  # chess symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
            "\U00002600-\U000026FF"  # misc symbols
            "]+",
            flags=re.UNICODE
        )

        def has_text_content(text):
            """Check if text has content beyond emojis and whitespace."""
            if not isinstance(text, str):
                return False
            # Remove emojis and whitespace
            cleaned = emoji_pattern.sub('', text).strip()
            # Check if any alphanumeric content remains
            return bool(re.search(r'[a-zA-Z0-9\u4e00-\u9fff]', cleaned))

        df = df[df['text'].apply(has_text_content)]

        removed = original_count - len(df)
        self.logger.info(f"Removed {removed} emoji-only comments ({len(df)} remaining)")
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

        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)

        # Step 4: Remove emoji-only comments
        df = self.remove_emoji_only(df)

        # Step 5: Clean text
        df = self.clean_text(df)

        # Step 6: Save to CSV
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
