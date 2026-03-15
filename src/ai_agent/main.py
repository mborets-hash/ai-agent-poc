"""Main entry point for the AI Investment Agent."""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ai_agent import config

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_banner() -> None:
    """Print the application banner."""
    banner = Text()
    banner.append("AI Investment Agent", style="bold cyan")
    banner.append("\n")
    banner.append("Powered by LangChain + LlamaIndex", style="dim")
    console.print(Panel(banner, title="[bold green]POC[/bold green]", expand=False))


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run a full portfolio analysis."""
    from ai_agent.agent.orchestrator import InvestmentAgent

    agent = InvestmentAgent(verbose=args.verbose)
    console.print("\n[bold yellow]Running comprehensive analysis...[/bold yellow]\n")

    result = agent.run_full_analysis(
        additional_instructions=args.instructions if args.instructions else ""
    )

    console.print(Panel(result, title="[bold green]Analysis Report[/bold green]"))
    console.print(
        "\n[dim italic]Disclaimer: This is for educational/informational purposes only. "
        "Not financial advice.[/dim italic]\n"
    )


def cmd_query(args: argparse.Namespace) -> None:
    """Ask a specific investment question."""
    from ai_agent.agent.orchestrator import InvestmentAgent

    agent = InvestmentAgent(verbose=args.verbose)
    question = " ".join(args.question)
    console.print(f"\n[bold yellow]Querying: {question}[/bold yellow]\n")

    result = agent.query(question)

    console.print(Panel(result, title="[bold green]Response[/bold green]"))
    console.print(
        "\n[dim italic]Disclaimer: This is for educational/informational purposes only. "
        "Not financial advice.[/dim italic]\n"
    )


def cmd_chat(args: argparse.Namespace) -> None:
    """Start an interactive chat session with the agent."""
    from ai_agent.agent.orchestrator import InvestmentAgent

    agent = InvestmentAgent(verbose=args.verbose)
    console.print("\n[bold green]Interactive mode. Type 'quit' or 'exit' to stop.[/bold green]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You>[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        console.print("[dim]Thinking...[/dim]")
        response = agent.chat(user_input)
        console.print(f"\n[bold green]Agent>[/bold green] {response}\n")


def cmd_news(args: argparse.Namespace) -> None:
    """Fetch and display financial news."""
    from ai_agent.tools.news_feed import analyze_news

    console.print("\n[bold yellow]Fetching financial news...[/bold yellow]\n")
    result = analyze_news(query=args.query if args.query else None)
    console.print(Panel(result, title="[bold green]News Feed[/bold green]"))


def cmd_portfolio(args: argparse.Namespace) -> None:
    """Display portfolio analysis."""
    from ai_agent.tools.portfolio import (
        compute_portfolio_summary,
        format_portfolio_analysis,
        load_portfolio,
    )

    console.print("\n[bold yellow]Analyzing portfolio...[/bold yellow]\n")
    holdings = load_portfolio(args.file if args.file else None)

    if not holdings:
        console.print("[red]No portfolio data found.[/red]")
        return

    summary = compute_portfolio_summary(holdings)
    result = format_portfolio_analysis(summary)
    console.print(Panel(result, title="[bold green]Portfolio Analysis[/bold green]"))


def cmd_market(args: argparse.Namespace) -> None:
    """Display market data."""
    from ai_agent.tools.market_data import fetch_market_data, format_market_data

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = config.DEFAULT_WATCHLIST
    console.print(f"\n[bold yellow]Fetching market data for: {', '.join(symbols)}[/bold yellow]\n")

    data = fetch_market_data(symbols)
    result = format_market_data(data)
    console.print(Panel(result, title="[bold green]Market Data[/bold green]"))


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Investment Agent - Portfolio analysis and recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run comprehensive portfolio analysis with AI recommendations"
    )
    analyze_parser.add_argument(
        "-i",
        "--instructions",
        type=str,
        default="",
        help="Additional instructions for the analysis",
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # query command
    query_parser = subparsers.add_parser("query", help="Ask a specific investment question")
    query_parser.add_argument("question", nargs="+", help="Your investment question")
    query_parser.set_defaults(func=cmd_query)

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat with the agent")
    chat_parser.set_defaults(func=cmd_chat)

    # news command
    news_parser = subparsers.add_parser("news", help="Fetch and display financial news")
    news_parser.add_argument(
        "-q", "--query", type=str, default="", help="Search query for the news"
    )
    news_parser.set_defaults(func=cmd_news)

    # portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", help="Display portfolio analysis")
    portfolio_parser.add_argument(
        "-f", "--file", type=str, default="", help="Path to portfolio JSON file"
    )
    portfolio_parser.set_defaults(func=cmd_portfolio)

    # market command
    market_parser = subparsers.add_parser("market", help="Display market data for symbols")
    market_parser.add_argument(
        "-s",
        "--symbols",
        type=str,
        default="",
        help="Comma-separated ticker symbols (e.g., AAPL,MSFT,GOOGL)",
    )
    market_parser.set_defaults(func=cmd_market)

    return parser


def main() -> None:
    """Main entry point."""
    print_banner()

    warnings = config.validate_config()
    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    setup_logging(verbose=args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
