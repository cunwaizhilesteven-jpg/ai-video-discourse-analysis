"""
Visualization Module - Comment Visualizer

Generates visualizations for sentiment analysis and linguistic complexity.
Outputs both PNG (static) and HTML (interactive) formats.
"""

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS


class CommentVisualizer:
    """Generates visualizations from analyzed comment data."""

    def __init__(self, input_path: str = "data/merged_data.csv",
                 output_dir: str = "output",
                 log_dir: str = "logs"):
        """
        Initialize the visualizer.

        Args:
            input_path: Path to analyzed CSV file
            output_dir: Base directory for output files
            log_dir: Directory for log files
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.png_dir = self.output_dir / "png"
        self.html_dir = self.output_dir / "html"
        self.log_dir = Path(log_dir)

        # Create directories
        self.png_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.df = None

        # Setup logging
        self._setup_logging()

        # Set style for matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

    def _setup_logging(self):
        """Configure logging to file and console."""
        log_file = self.log_dir / f"visualizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load the analyzed data."""
        self.logger.info(f"Loading data from: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        self.logger.info(f"Loaded {len(self.df)} comments")

    def save_png(self, fig, name: str):
        """Save matplotlib figure as PNG."""
        path = self.png_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        self.logger.info(f"Saved PNG: {path}")
        plt.close(fig)

    def save_html(self, fig, name: str):
        """Save plotly figure as HTML."""
        path = self.html_dir / f"{name}.html"
        fig.write_html(str(path))
        self.logger.info(f"Saved HTML: {path}")

    def plot_sentiment_pie(self):
        """
        Chart 1: Sentiment distribution pie chart.

        Returns:
            Tuple of (matplotlib_fig, plotly_fig)
        """
        self.logger.info("Generating sentiment distribution pie chart...")

        # Calculate sentiment counts
        sentiment_counts = self.df['sentiment_label'].value_counts()

        # Colors
        colors_mpl = ['#2ecc71', '#e74c3c', '#95a5a6']  # green, red, gray
        colors_plotly = ['#2ecc71', '#e74c3c', '#95a5a6']

        # Matplotlib version
        fig_mpl, ax = plt.subplots(figsize=(10, 8))

        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=colors_mpl[:len(sentiment_counts)],
            explode=[0.02] * len(sentiment_counts),
            shadow=True,
            startangle=90
        )

        ax.set_title('Sentiment Distribution of YouTube Comments\non AI-Generated Videos',
                     fontsize=14, fontweight='bold')

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        self.save_png(fig_mpl, "sentiment_distribution")

        # Plotly version (interactive)
        fig_plotly = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index.tolist(),
            values=sentiment_counts.values.tolist(),
            hole=0.3,
            marker_colors=colors_plotly[:len(sentiment_counts)],
            textinfo='label+percent',
            textfont_size=14,
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
        )])

        fig_plotly.update_layout(
            title={
                'text': 'Sentiment Distribution of YouTube Comments<br>on AI-Generated Videos',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )

        self.save_html(fig_plotly, "sentiment_distribution")

        return fig_mpl, fig_plotly

    def plot_engagement_scatter(self):
        """
        Chart 2: Scatter plot - X: Lexical Density (Engagement), Y: Sentiment Score.

        Returns:
            Tuple of (matplotlib_fig, plotly_fig)
        """
        self.logger.info("Generating engagement scatter plot...")

        # Prepare data
        df_plot = self.df.copy()

        # Convert sentiment to signed score (positive = +score, negative = -score)
        df_plot['signed_sentiment'] = df_plot.apply(
            lambda row: row['sentiment_score'] if row['sentiment_label'] == 'POSITIVE'
            else -row['sentiment_score'],
            axis=1
        )

        # Matplotlib version
        fig_mpl, ax = plt.subplots(figsize=(12, 8))

        # Color by sentiment label
        colors = df_plot['sentiment_label'].map({
            'POSITIVE': '#2ecc71',
            'NEGATIVE': '#e74c3c',
            'UNKNOWN': '#95a5a6'
        })

        scatter = ax.scatter(
            df_plot['lexical_density'],
            df_plot['signed_sentiment'],
            c=colors,
            alpha=0.5,
            s=30
        )

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Lexical Density (Engagement Depth)', fontsize=12)
        ax.set_ylabel('Sentiment Score (Acceptance)', fontsize=12)
        ax.set_title('User Engagement vs. Acceptance of AI-Generated Videos',
                     fontsize=14, fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Positive'),
            Patch(facecolor='#e74c3c', label='Negative')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        self.save_png(fig_mpl, "engagement_scatter")

        # Plotly version (interactive)
        fig_plotly = px.scatter(
            df_plot,
            x='lexical_density',
            y='signed_sentiment',
            color='sentiment_label',
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEGATIVE': '#e74c3c',
                'UNKNOWN': '#95a5a6'
            },
            hover_data=['text', 'votes', 'word_count'],
            opacity=0.6,
            title='User Engagement vs. Acceptance of AI-Generated Videos'
        )

        fig_plotly.update_layout(
            xaxis_title='Lexical Density (Engagement Depth)',
            yaxis_title='Sentiment Score (Acceptance)',
            title={
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            }
        )

        fig_plotly.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        self.save_html(fig_plotly, "engagement_scatter")

        return fig_mpl, fig_plotly

    def plot_negative_wordcloud(self):
        """
        Chart 3: Word cloud of negative comments.

        Returns:
            WordCloud image (PNG only, no interactive version)
        """
        self.logger.info("Generating negative comments word cloud...")

        # Filter negative comments
        negative_comments = self.df[self.df['sentiment_label'] == 'NEGATIVE']['text']

        if len(negative_comments) == 0:
            self.logger.warning("No negative comments found for word cloud")
            return None

        # Combine all negative comments
        text = ' '.join(negative_comments.astype(str).tolist())

        # Extended stopwords
        stopwords = set(STOPWORDS)
        stopwords.update([
            'video', 'videos', 'youtube', 'channel', 'subscribe', 'like',
            'comment', 'watch', 'watching', 'watched', 'will', 'would',
            'could', 'should', 'one', 'two', 'three', 'first', 'second',
            'thing', 'things', 'really', 'much', 'many', 'make', 'made',
            'see', 'know', 'think', 'want', 'get', 'got', 'going', 'go'
        ])

        # Generate word cloud
        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color='white',
            stopwords=stopwords,
            max_words=100,
            colormap='Reds',
            collocations=False,
            min_font_size=10
        ).generate(text)

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Most Common Words in Negative Comments\nabout AI-Generated Videos',
                     fontsize=16, fontweight='bold', pad=20)

        self.save_png(fig, "negative_wordcloud")

        return wordcloud

    def run(self):
        """
        Generate all visualizations.
        """
        self.logger.info("=" * 50)
        self.logger.info("Starting Visualization Generation")
        self.logger.info("=" * 50)

        # Load data
        self.load_data()

        # Check for required columns
        required_cols = ['sentiment_label', 'sentiment_score', 'lexical_density', 'text']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            self.logger.error("Please run analyzer.py first")
            return

        # Generate visualizations
        self.plot_sentiment_pie()
        self.plot_engagement_scatter()
        self.plot_negative_wordcloud()

        # Summary
        self.logger.info("=" * 50)
        self.logger.info("Visualization Complete!")
        self.logger.info(f"PNG files saved to: {self.png_dir}")
        self.logger.info(f"HTML files saved to: {self.html_dir}")
        self.logger.info("Generated charts:")
        self.logger.info("  1. sentiment_distribution (PNG + HTML)")
        self.logger.info("  2. engagement_scatter (PNG + HTML)")
        self.logger.info("  3. negative_wordcloud (PNG only)")
        self.logger.info("=" * 50)


def main():
    """Entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualizations from analyzed comments")
    parser.add_argument("--input", default="data/merged_data.csv", help="Input CSV path")
    parser.add_argument("--output-dir", default="output", help="Output directory")

    args = parser.parse_args()

    visualizer = CommentVisualizer(
        input_path=args.input,
        output_dir=args.output_dir
    )
    visualizer.run()


if __name__ == "__main__":
    main()
