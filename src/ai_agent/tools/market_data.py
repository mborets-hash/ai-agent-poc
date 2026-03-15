"""Market data analysis tool using yfinance."""

from __future__ import annotations

import logging

import yfinance as yf
from langchain_core.tools import tool

from ai_agent import config
from ai_agent.models.schemas import MarketData

logger = logging.getLogger(__name__)


def fetch_market_data(symbols: list[str]) -> list[MarketData]:
    """Fetch current market data for given symbols using yfinance.

    Args:
        symbols: List of ticker symbols to fetch data for.

    Returns:
        List of MarketData objects with current market information.
    """
    results: list[MarketData] = []

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "currentPrice" not in info and "regularMarketPrice" not in info:
                fast_info = ticker.fast_info
                current_price = getattr(fast_info, "last_price", 0.0) or 0.0
                previous_close = getattr(fast_info, "previous_close", None)
                market_cap = getattr(fast_info, "market_cap", None)

                market_data = MarketData(
                    symbol=symbol.upper(),
                    name=symbol.upper(),
                    current_price=current_price,
                    previous_close=previous_close,
                    day_change=(current_price - previous_close if previous_close else None),
                    day_change_pct=(
                        ((current_price - previous_close) / previous_close * 100)
                        if previous_close
                        else None
                    ),
                    market_cap=market_cap,
                )
            else:
                current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0.0)
                previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

                market_data = MarketData(
                    symbol=symbol.upper(),
                    name=info.get("shortName", info.get("longName", symbol.upper())),
                    current_price=current_price,
                    previous_close=previous_close,
                    day_change=(current_price - previous_close if previous_close else None),
                    day_change_pct=(
                        ((current_price - previous_close) / previous_close * 100)
                        if previous_close
                        else None
                    ),
                    volume=info.get("volume") or info.get("regularMarketVolume"),
                    market_cap=info.get("marketCap"),
                    pe_ratio=info.get("trailingPE") or info.get("forwardPE"),
                    fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                    fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                    dividend_yield=info.get("dividendYield"),
                )

            results.append(market_data)
            logger.info("Fetched market data for %s: $%.2f", symbol, current_price)

        except Exception:
            logger.exception("Error fetching market data for %s", symbol)

    return results


def format_market_data(data_list: list[MarketData]) -> str:
    """Format market data as a readable string.

    Args:
        data_list: List of MarketData objects.

    Returns:
        Formatted market data string.
    """
    if not data_list:
        return "No market data available."

    lines: list[str] = [
        "=" * 80,
        "MARKET DATA OVERVIEW",
        "=" * 80,
    ]

    for data in data_list:
        change_str = ""
        if data.day_change is not None and data.day_change_pct is not None:
            direction = "+" if data.day_change >= 0 else ""
            change_str = (
                f"  Change: {direction}${data.day_change:.2f}"
                f" ({direction}{data.day_change_pct:.2f}%)"
            )

        lines.append(f"\n  {data.symbol} - {data.name}")
        lines.append(f"    Price: ${data.current_price:,.2f}{change_str}")

        details: list[str] = []
        if data.market_cap:
            if data.market_cap >= 1e12:
                details.append(f"Mkt Cap: ${data.market_cap / 1e12:.2f}T")
            elif data.market_cap >= 1e9:
                details.append(f"Mkt Cap: ${data.market_cap / 1e9:.2f}B")
            else:
                details.append(f"Mkt Cap: ${data.market_cap / 1e6:.2f}M")
        if data.pe_ratio:
            details.append(f"P/E: {data.pe_ratio:.2f}")
        if data.dividend_yield:
            details.append(f"Div Yield: {data.dividend_yield * 100:.2f}%")
        if data.volume:
            details.append(f"Volume: {data.volume:,}")
        if data.fifty_two_week_high and data.fifty_two_week_low:
            details.append(
                f"52W Range: ${data.fifty_two_week_low:.2f} - ${data.fifty_two_week_high:.2f}"
            )

        if details:
            lines.append(f"    {' | '.join(details)}")

    return "\n".join(lines)


@tool
def get_market_data_tool(symbols: str = "") -> str:
    """Fetch current market data for financial instruments.

    Retrieves real-time market data including price, change, market cap, P/E ratio,
    dividend yield, volume, and 52-week range for the specified ticker symbols.

    Args:
        symbols: Comma-separated ticker symbols (e.g., "AAPL,MSFT,GOOGL").
            If empty, uses the default watchlist from configuration.

    Returns:
        A formatted string with current market data for each symbol.
    """
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        symbol_list = config.DEFAULT_WATCHLIST

    data = fetch_market_data(symbol_list)
    return format_market_data(data)
