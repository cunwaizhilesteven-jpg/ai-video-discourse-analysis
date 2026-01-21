"""
Analysis Module - Sentiment and Linguistic Complexity Analyzer

Analyzes comments using DistilBERT for sentiment and textstat for linguistic complexity.
"""

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import textstat
from tqdm import tqdm

# Transformers imports
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


class CommentAnalyzer:
    """Analyzes comments for sentiment and linguistic complexity."""

    def __init__(self, input_path: str = "data/merged_data.csv",
                 output_path: str = "data/merged_data.csv",
                 log_dir: str = "logs",
                 batch_size: int = 32):
        """
        Initialize the analyzer.

        Args:
            input_path: Path to input CSV file
            output_path: Path for output CSV file (can be same as input)
            log_dir: Directory for log files
            batch_size: Batch size for sentiment analysis
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.log_dir = Path(log_dir)
        self.batch_size = batch_size

        # Model components (loaded lazily)
        self.sentiment_pipeline = None
        self.device = None

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging to file and console."""
        log_file = self.log_dir / f"analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load the DistilBERT sentiment analysis model."""
        self.logger.info("Loading DistilBERT sentiment model...")

        # Check for GPU availability
        self.device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if self.device == 0 else "CPU"
        self.logger.info(f"Using device: {device_name}")

        # Load the sentiment analysis pipeline
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            truncation=True,
            max_length=512
        )

        self.logger.info("Model loaded successfully")

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment for a single text.

        Args:
            text: Input text

        Returns:
            Dict with 'label' (POSITIVE/NEGATIVE) and 'score' (confidence)
        """
        if not self.sentiment_pipeline:
            self.load_model()

        try:
            result = self.sentiment_pipeline(text[:512])[0]  # Truncate to max length
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            self.logger.warning(f"Error analyzing sentiment: {e}")
            return {'label': 'UNKNOWN', 'score': 0.0}

    def analyze_sentiment_batch(self, texts: list) -> list:
        """
        Analyze sentiment for multiple texts in batches.

        Args:
            texts: List of text strings

        Returns:
            List of dicts with 'label' and 'score'
        """
        if not self.sentiment_pipeline:
            self.load_model()

        results = []
        self.logger.info(f"Analyzing sentiment for {len(texts)} comments...")

        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Sentiment Analysis"):
            batch = texts[i:i + self.batch_size]

            # Truncate texts to max length
            batch = [text[:512] if isinstance(text, str) else "" for text in batch]

            try:
                batch_results = self.sentiment_pipeline(batch)
                for result in batch_results:
                    results.append({
                        'label': result['label'],
                        'score': result['score']
                    })
            except Exception as e:
                self.logger.warning(f"Error in batch {i}: {e}")
                # Fill with unknown for failed batch
                for _ in batch:
                    results.append({'label': 'UNKNOWN', 'score': 0.0})

        return results

    def calculate_complexity(self, text: str) -> dict:
        """
        Calculate linguistic complexity metrics using textstat.

        Args:
            text: Input text

        Returns:
            Dict with complexity metrics
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                'lexical_density': 0.0,
                'flesch_reading_ease': 0.0,
                'word_count': 0
            }

        try:
            # Lexical density: ratio of content words to total words
            # Using textstat's lexicon count as a proxy
            word_count = textstat.lexicon_count(text, removepunct=True)

            # Flesch Reading Ease (higher = easier to read)
            flesch = textstat.flesch_reading_ease(text)

            # Calculate lexical density as unique words / total words
            words = text.lower().split()
            unique_words = set(words)
            lexical_density = len(unique_words) / len(words) if words else 0.0

            return {
                'lexical_density': round(lexical_density, 4),
                'flesch_reading_ease': round(flesch, 2),
                'word_count': word_count
            }
        except Exception as e:
            self.logger.warning(f"Error calculating complexity: {e}")
            return {
                'lexical_density': 0.0,
                'flesch_reading_ease': 0.0,
                'word_count': 0
            }

    def run(self) -> pd.DataFrame:
        """
        Main execution logic.

        Returns:
            DataFrame with analysis columns added
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting Comment Analysis")
        self.logger.info("=" * 50)

        # Load data
        self.logger.info(f"Loading data from: {self.input_path}")
        df = pd.read_csv(self.input_path)
        self.logger.info(f"Loaded {len(df)} comments")

        # Sentiment Analysis (batch processing)
        texts = df['text'].tolist()
        sentiment_results = self.analyze_sentiment_batch(texts)

        df['sentiment_label'] = [r['label'] for r in sentiment_results]
        df['sentiment_score'] = [r['score'] for r in sentiment_results]

        # Linguistic Complexity Analysis
        self.logger.info("Calculating linguistic complexity...")
        complexity_results = []
        for text in tqdm(texts, desc="Complexity Analysis"):
            complexity_results.append(self.calculate_complexity(text))

        df['lexical_density'] = [r['lexical_density'] for r in complexity_results]
        df['flesch_reading_ease'] = [r['flesch_reading_ease'] for r in complexity_results]
        df['word_count'] = [r['word_count'] for r in complexity_results]

        # Save results
        df.to_csv(self.output_path, index=False, encoding='utf-8')
        self.logger.info(f"Saved analyzed data to {self.output_path}")

        # Summary statistics
        self.logger.info("=" * 50)
        self.logger.info("Analysis Complete!")
        self.logger.info(f"Total comments analyzed: {len(df)}")

        sentiment_counts = df['sentiment_label'].value_counts()
        for label, count in sentiment_counts.items():
            pct = count / len(df) * 100
            self.logger.info(f"  {label}: {count} ({pct:.1f}%)")

        self.logger.info(f"Average lexical density: {df['lexical_density'].mean():.4f}")
        self.logger.info(f"Average word count: {df['word_count'].mean():.1f}")
        self.logger.info("=" * 50)

        return df


def main():
    """Entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze comments for sentiment and complexity")
    parser.add_argument("--input", default="data/merged_data.csv", help="Input CSV path")
    parser.add_argument("--output", default="data/merged_data.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for sentiment analysis")

    args = parser.parse_args()

    analyzer = CommentAnalyzer(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size
    )
    analyzer.run()


if __name__ == "__main__":
    main()
