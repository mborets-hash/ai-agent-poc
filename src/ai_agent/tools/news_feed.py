"""News feed analysis tool using RSS feeds and LlamaIndex indexing."""

from __future__ import annotations

import logging
from datetime import datetime

import feedparser
from langchain_core.tools import tool

from ai_agent import config
from ai_agent.indexing.document_store import DocumentStore
from ai_agent.models.schemas import NewsArticle

logger = logging.getLogger(__name__)


def fetch_news_articles(
    feed_urls: list[str] | None = None,
    max_articles_per_feed: int = 10,
) -> list[NewsArticle]:
    """Fetch news articles from RSS feeds.

    Args:
        feed_urls: List of RSS feed URLs. Defaults to configured feeds.
        max_articles_per_feed: Maximum articles to fetch per feed.

    Returns:
        List of parsed NewsArticle objects.
    """
    urls = feed_urls or config.NEWS_FEEDS
    articles: list[NewsArticle] = []

    for url in urls:
        try:
            feed = feedparser.parse(url)
            source_name = feed.feed.get("title", url)

            for entry in feed.entries[:max_articles_per_feed]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6])
                    except (TypeError, ValueError):
                        pass

                article = NewsArticle(
                    title=entry.get("title", "Untitled"),
                    summary=entry.get("summary", entry.get("description", "")),
                    url=entry.get("link", ""),
                    source=source_name,
                    published=published,
                )
                articles.append(article)

            num_fetched = len(feed.entries[:max_articles_per_feed])
            logger.info("Fetched %d articles from %s", num_fetched, source_name)
        except Exception:
            logger.exception("Error fetching feed %s", url)

    return articles


def analyze_news(
    query: str | None = None,
    symbols: list[str] | None = None,
) -> str:
    """Fetch news, index with LlamaIndex, and optionally query.

    Args:
        query: Optional query to search indexed news.
        symbols: Optional list of symbols to focus on.

    Returns:
        Formatted string with news analysis results.
    """
    articles = fetch_news_articles()

    if not articles:
        return "No news articles could be fetched from configured feeds."

    if symbols:
        symbol_set = {s.upper() for s in symbols}
        filtered = [
            a
            for a in articles
            if any(s in a.title.upper() or s in a.summary.upper() for s in symbol_set)
        ]
        if filtered:
            articles = filtered

    doc_store = DocumentStore()
    indexed_count = doc_store.index_news_articles(articles)

    lines: list[str] = [f"Fetched and indexed {indexed_count} news articles.\n"]

    if query:
        result = doc_store.query(query)
        lines.append(f"Query: {query}")
        lines.append(f"Answer: {result}\n")

    lines.append("Recent Headlines:")
    for article in articles[:15]:
        date_str = article.published.strftime("%Y-%m-%d") if article.published else "N/A"
        lines.append(f"  [{date_str}] {article.title} ({article.source})")
        if article.url:
            lines.append(f"    URL: {article.url}")

    return "\n".join(lines)


@tool
def get_news_feed_tool(query: str = "") -> str:
    """Fetch and analyze financial news from RSS feeds.

    Fetches news from multiple financial news sources, indexes them for semantic
    search using LlamaIndex, and returns recent headlines and optionally answers
    a specific query about the news.

    Args:
        query: Optional natural language query to search the news.
            If empty, returns recent headlines only.

    Returns:
        A formatted string with news headlines and optional query results.
    """
    symbols_from_query: list[str] = []
    if query:
        import re

        ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")
        potential_tickers = ticker_pattern.findall(query)
        common_words = {
            "THE",
            "AND",
            "FOR",
            "ARE",
            "BUT",
            "NOT",
            "YOU",
            "ALL",
            "CAN",
            "HAS",
            "HER",
            "WAS",
            "ONE",
            "OUR",
            "OUT",
            "HOW",
            "WHO",
            "WHAT",
            "WHEN",
            "WHY",
            "ANY",
            "NEW",
            "NOW",
            "OLD",
            "SEE",
            "WAY",
            "MAY",
            "SAY",
            "SHE",
            "TWO",
            "USE",
            "BOY",
            "DID",
            "GET",
            "HIM",
            "HIS",
            "LET",
            "PUT",
            "TOP",
            "TOO",
        }
        symbols_from_query = [t for t in potential_tickers if t not in common_words]

    return analyze_news(query=query if query else None, symbols=symbols_from_query or None)
