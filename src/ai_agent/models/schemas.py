"""Pydantic data models for portfolio, news, and market data."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class AssetType(StrEnum):
    """Types of financial assets."""

    STOCK = "stock"
    ETF = "etf"
    BOND = "bond"
    CRYPTO = "crypto"
    MUTUAL_FUND = "mutual_fund"
    OTHER = "other"


class RecommendationAction(StrEnum):
    """Possible recommendation actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    INCREASE = "increase"
    DECREASE = "decrease"


class PortfolioHolding(BaseModel):
    """Represents a single holding in a portfolio."""

    symbol: str = Field(..., description="Ticker symbol")
    name: str = Field(..., description="Full name of the asset")
    asset_type: AssetType = Field(default=AssetType.STOCK, description="Type of asset")
    quantity: float = Field(..., description="Number of shares/units held")
    average_cost: float = Field(..., description="Average cost per share/unit")
    current_price: float | None = Field(default=None, description="Current market price")
    currency: str = Field(default="USD", description="Currency of the holding")

    @property
    def total_cost(self) -> float:
        """Total cost basis of the holding."""
        return self.quantity * self.average_cost

    @property
    def total_value(self) -> float | None:
        """Current total value of the holding."""
        if self.current_price is None:
            return None
        return self.quantity * self.current_price

    @property
    def gain_loss(self) -> float | None:
        """Unrealized gain/loss."""
        if self.total_value is None:
            return None
        return self.total_value - self.total_cost

    @property
    def gain_loss_pct(self) -> float | None:
        """Unrealized gain/loss percentage."""
        if self.gain_loss is None or self.total_cost == 0:
            return None
        return (self.gain_loss / self.total_cost) * 100


class PortfolioSummary(BaseModel):
    """Summary of the entire portfolio."""

    holdings: list[PortfolioHolding] = Field(default_factory=list)
    total_value: float = Field(default=0.0, description="Total portfolio value")
    total_cost: float = Field(default=0.0, description="Total cost basis")
    total_gain_loss: float = Field(default=0.0, description="Total unrealized gain/loss")
    total_gain_loss_pct: float = Field(default=0.0, description="Total gain/loss percentage")
    last_updated: datetime = Field(default_factory=datetime.now)


class NewsArticle(BaseModel):
    """Represents a parsed news article."""

    title: str = Field(..., description="Article title")
    summary: str = Field(default="", description="Article summary or description")
    url: str = Field(default="", description="URL to the full article")
    source: str = Field(default="", description="News source name")
    published: datetime | None = Field(default=None, description="Publication date")
    related_symbols: list[str] = Field(default_factory=list, description="Ticker symbols mentioned")


class MarketData(BaseModel):
    """Market data for a financial instrument."""

    symbol: str = Field(..., description="Ticker symbol")
    name: str = Field(default="", description="Instrument name")
    current_price: float = Field(..., description="Current price")
    previous_close: float | None = Field(default=None, description="Previous closing price")
    day_change: float | None = Field(default=None, description="Day change in price")
    day_change_pct: float | None = Field(default=None, description="Day change percentage")
    volume: int | None = Field(default=None, description="Trading volume")
    market_cap: float | None = Field(default=None, description="Market capitalization")
    pe_ratio: float | None = Field(default=None, description="P/E ratio")
    fifty_two_week_high: float | None = Field(default=None, description="52-week high")
    fifty_two_week_low: float | None = Field(default=None, description="52-week low")
    dividend_yield: float | None = Field(default=None, description="Dividend yield")


class Recommendation(BaseModel):
    """An investment recommendation."""

    symbol: str = Field(..., description="Ticker symbol")
    action: RecommendationAction = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence level (0-1)")
    reasoning: str = Field(..., description="Reasoning behind the recommendation")
    target_allocation_pct: float | None = Field(
        default=None, description="Suggested portfolio allocation percentage"
    )


class AnalysisReport(BaseModel):
    """Complete analysis report with recommendations."""

    portfolio_summary: PortfolioSummary | None = Field(default=None)
    news_highlights: list[NewsArticle] = Field(default_factory=list)
    market_overview: list[MarketData] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    analysis_summary: str = Field(default="", description="Overall analysis summary")
    generated_at: datetime = Field(default_factory=datetime.now)
