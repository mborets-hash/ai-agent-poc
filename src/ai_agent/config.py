"""Configuration management for the AI investment agent."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_STORAGE_DIR = PROJECT_ROOT / ".index_storage"

# OpenAI Configuration
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))

# Portfolio Configuration
PORTFOLIO_FILE: str = os.getenv("PORTFOLIO_FILE", str(DATA_DIR / "sample_portfolio.json"))

# News Feed Configuration
NEWS_FEEDS: list[str] = [
    url.strip()
    for url in os.getenv(
        "NEWS_FEEDS",
        ",".join(
            [
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
                "https://www.investing.com/rss/news.rss",
                "https://feeds.marketwatch.com/marketwatch/topstories/",
            ]
        ),
    ).split(",")
    if url.strip()
]

# Market Data Configuration
DEFAULT_WATCHLIST: list[str] = [
    s.strip()
    for s in os.getenv(
        "WATCHLIST",
        "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,SPY,QQQ,BND",
    ).split(",")
    if s.strip()
]

# Agent Configuration
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "15"))
VERBOSE_AGENT: bool = os.getenv("VERBOSE_AGENT", "true").lower() == "true"


def validate_config() -> list[str]:
    """Validate configuration and return list of warnings."""
    warnings: list[str] = []

    if not OPENAI_API_KEY:
        warnings.append(
            "OPENAI_API_KEY is not set. The agent will not be able to use LLM capabilities."
        )

    portfolio_path = Path(PORTFOLIO_FILE)
    if not portfolio_path.exists():
        warnings.append(f"Portfolio file not found at {PORTFOLIO_FILE}.")

    return warnings
