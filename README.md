# AI Investment Agent POC

An AI-powered investment analysis agent built with **LangChain** and **LlamaIndex** that analyzes news feeds, market data, and your existing portfolio to provide actionable investment recommendations.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLI Interface                      │
│              (main.py - Rich console)                │
├─────────────────────────────────────────────────────┤
│              LangChain Agent Orchestrator             │
│     (Ollama Tool Calling Agent + AgentExecutor)        │
├──────────┬──────────────┬───────────────────────────┤
│  Portfolio│   News Feed   │      Market Data          │
│  Analysis │   Analysis    │      Analysis             │
│  Tool     │   Tool        │      Tool                 │
│           │               │                           │
│  JSON     │  RSS/Feedparser│     yfinance             │
│  loader   │  + LlamaIndex │     real-time data        │
│           │  indexing      │                           │
└──────────┴──────────────┴───────────────────────────┘
```

### Components

- **LangChain Agent**: Orchestrates the analysis workflow using Ollama tool calling. Decides which tools to use and synthesizes results into recommendations. Runs entirely locally.
- **LlamaIndex Document Store**: Indexes news articles into a vector store for semantic search and retrieval-augmented generation (RAG).
- **Portfolio Tool**: Loads holdings from JSON, computes allocation, gain/loss, and performance metrics.
- **News Feed Tool**: Fetches financial news via RSS (Yahoo Finance, MarketWatch, etc.), indexes with LlamaIndex for semantic querying.
- **Market Data Tool**: Retrieves real-time market data via yfinance (price, P/E, market cap, dividends, 52-week range).

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally (`ollama serve`)
- Models pulled: `ollama pull llama3.1` and `ollama pull nomic-embed-text`

### Installation

```bash
# Clone the repository
git clone https://github.com/mborets-hash/ai-agent-poc.git
cd ai-agent-poc

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package
pip install -e ".[dev]"

# Pull required Ollama models
ollama pull llama3.1
ollama pull nomic-embed-text

# Configure environment (optional - defaults work with local Ollama)
cp .env.example .env
```

## Usage

### Full Analysis

Run a comprehensive analysis of your portfolio with AI-driven recommendations:

```bash
ai-agent analyze
ai-agent analyze -i "Focus on tech sector exposure and risk"
```

### Ask Questions

Ask specific investment questions:

```bash
ai-agent query "Should I increase my NVDA position?"
ai-agent query "What is my portfolio's sector concentration risk?"
```

### Interactive Chat

Start an interactive session:

```bash
ai-agent chat
```

### Individual Tools

Use tools independently without the AI agent:

```bash
# View portfolio analysis
ai-agent portfolio

# Fetch financial news
ai-agent news
ai-agent news -q "AI semiconductor stocks"

# Get market data
ai-agent market
ai-agent market -s "AAPL,NVDA,TSLA"
```

### Verbose Mode

Add `-v` for detailed logging:

```bash
ai-agent -v analyze
```

## Configuration

Configuration is managed via environment variables (`.env` file):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1` | LLM model for the agent |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for LlamaIndex |
| `OLLAMA_TEMPERATURE` | `0.1` | LLM temperature |
| `OLLAMA_REQUEST_TIMEOUT` | `120.0` | Request timeout in seconds |
| `PORTFOLIO_FILE` | `data/sample_portfolio.json` | Path to portfolio JSON |
| `NEWS_FEEDS` | Yahoo Finance, MarketWatch | Comma-separated RSS feed URLs |
| `WATCHLIST` | Top tech + index ETFs | Comma-separated ticker symbols |
| `MAX_AGENT_ITERATIONS` | `15` | Max agent reasoning iterations |
| `VERBOSE_AGENT` | `true` | Show agent reasoning steps |

## Portfolio File Format

```json
{
  "holdings": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "asset_type": "stock",
      "quantity": 50,
      "average_cost": 150.00,
      "current_price": 195.00,
      "currency": "USD"
    }
  ]
}
```

Supported `asset_type` values: `stock`, `etf`, `bond`, `crypto`, `mutual_fund`, `other`.

## Project Structure

```
ai-agent-poc/
├── pyproject.toml              # Project config and dependencies
├── .env.example                # Environment variable template
├── data/
│   └── sample_portfolio.json   # Sample portfolio for demo
├── src/ai_agent/
│   ├── main.py                 # CLI entry point
│   ├── config.py               # Configuration management
│   ├── agent/
│   │   ├── orchestrator.py     # LangChain agent orchestrator
│   │   └── prompts.py          # System and analysis prompts
│   ├── tools/
│   │   ├── news_feed.py        # RSS news fetching + LlamaIndex indexing
│   │   ├── portfolio.py        # Portfolio loading and analysis
│   │   └── market_data.py      # yfinance market data retrieval
│   ├── indexing/
│   │   └── document_store.py   # LlamaIndex vector store management
│   └── models/
│       └── schemas.py          # Pydantic data models
└── tests/
```

## Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.
