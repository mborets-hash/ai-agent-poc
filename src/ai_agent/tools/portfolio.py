"""Portfolio analysis tool for loading and analyzing investment holdings."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.tools import tool

from ai_agent import config
from ai_agent.models.schemas import AssetType, PortfolioHolding, PortfolioSummary

logger = logging.getLogger(__name__)


def load_portfolio(file_path: str | Path | None = None) -> list[PortfolioHolding]:
    """Load portfolio holdings from a JSON file.

    Args:
        file_path: Path to portfolio JSON file. Defaults to configured path.

    Returns:
        List of PortfolioHolding objects.
    """
    path = Path(file_path) if file_path else Path(config.PORTFOLIO_FILE)

    if not path.exists():
        logger.warning("Portfolio file not found: %s", path)
        return []

    with open(path) as f:
        data = json.load(f)

    holdings: list[PortfolioHolding] = []
    items = data if isinstance(data, list) else data.get("holdings", [])

    for item in items:
        if "asset_type" in item and isinstance(item["asset_type"], str):
            try:
                item["asset_type"] = AssetType(item["asset_type"])
            except ValueError:
                item["asset_type"] = AssetType.OTHER
        holdings.append(PortfolioHolding(**item))

    return holdings


def compute_portfolio_summary(holdings: list[PortfolioHolding]) -> PortfolioSummary:
    """Compute portfolio summary statistics.

    Args:
        holdings: List of portfolio holdings.

    Returns:
        PortfolioSummary with aggregated metrics.
    """
    total_value = 0.0
    total_cost = 0.0

    for h in holdings:
        total_cost += h.total_cost
        if h.total_value is not None:
            total_value += h.total_value
        else:
            total_value += h.total_cost

    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost * 100) if total_cost > 0 else 0.0

    return PortfolioSummary(
        holdings=holdings,
        total_value=total_value,
        total_cost=total_cost,
        total_gain_loss=total_gain_loss,
        total_gain_loss_pct=total_gain_loss_pct,
    )


def format_portfolio_analysis(summary: PortfolioSummary) -> str:
    """Format portfolio analysis as a readable string.

    Args:
        summary: Portfolio summary to format.

    Returns:
        Formatted portfolio analysis string.
    """
    lines: list[str] = [
        "=" * 60,
        "PORTFOLIO ANALYSIS",
        "=" * 60,
        f"Total Value:      ${summary.total_value:>12,.2f}",
        f"Total Cost Basis: ${summary.total_cost:>12,.2f}",
        f"Total Gain/Loss:  ${summary.total_gain_loss:>12,.2f}"
        f" ({summary.total_gain_loss_pct:+.2f}%)",
        "",
        "HOLDINGS BREAKDOWN:",
        "-" * 60,
    ]

    for h in summary.holdings:
        value = h.total_value if h.total_value is not None else h.total_cost
        weight = (value / summary.total_value * 100) if summary.total_value > 0 else 0

        gl_str = ""
        if h.gain_loss is not None and h.gain_loss_pct is not None:
            gl_str = f"  G/L: ${h.gain_loss:>10,.2f} ({h.gain_loss_pct:+.2f}%)"

        lines.append(
            f"  {h.symbol:<8} {h.name:<25} "
            f"Qty: {h.quantity:>8.2f}  "
            f"Value: ${value:>10,.2f}  "
            f"Weight: {weight:>5.1f}%"
            f"{gl_str}"
        )

    # Allocation by asset type
    type_allocation: dict[str, float] = {}
    for h in summary.holdings:
        value = h.total_value if h.total_value is not None else h.total_cost
        asset_type = h.asset_type.value
        type_allocation[asset_type] = type_allocation.get(asset_type, 0.0) + value

    lines.extend(["", "ALLOCATION BY ASSET TYPE:", "-" * 40])
    for asset_type, value in sorted(type_allocation.items(), key=lambda x: -x[1]):
        pct = (value / summary.total_value * 100) if summary.total_value > 0 else 0
        lines.append(f"  {asset_type:<15} ${value:>12,.2f}  ({pct:.1f}%)")

    return "\n".join(lines)


@tool
def get_portfolio_tool(file_path: str = "") -> str:
    """Analyze the current investment portfolio.

    Loads portfolio holdings from a JSON file and provides a detailed analysis
    including total value, cost basis, gain/loss, individual holdings breakdown,
    and allocation by asset type.

    Args:
        file_path: Optional path to a portfolio JSON file.
            If empty, uses the default configured portfolio file.

    Returns:
        A formatted string with portfolio analysis including holdings,
        allocation, and performance metrics.
    """
    path = file_path if file_path else None
    holdings = load_portfolio(path)

    if not holdings:
        return (
            "No portfolio data found. Please ensure a portfolio JSON file exists at "
            f"{config.PORTFOLIO_FILE} or provide a valid file path."
        )

    summary = compute_portfolio_summary(holdings)
    return format_portfolio_analysis(summary)
