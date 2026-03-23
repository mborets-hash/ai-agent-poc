"""LangChain-based agent orchestrator for investment analysis using local Ollama."""

from __future__ import annotations

import logging

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph

from ai_agent import config
from ai_agent.agent.prompts import (
    ANALYSIS_PROMPT_TEMPLATE,
    QUERY_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from ai_agent.tools import get_market_data_tool, get_news_feed_tool, get_portfolio_tool

logger = logging.getLogger(__name__)


def _extract_response(result: dict) -> str:
    """Extract the final text response from the agent result."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


class InvestmentAgent:
    """AI Investment Analyst Agent powered by LangChain + Ollama.

    Orchestrates multiple tools (portfolio analysis, news feeds, market data)
    to provide comprehensive investment analysis and recommendations.
    Uses a local Ollama server for LLM inference.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        verbose: bool | None = None,
    ) -> None:
        self._model = model or config.OLLAMA_MODEL
        self._temperature = temperature if temperature is not None else config.OLLAMA_TEMPERATURE
        self._verbose = verbose if verbose is not None else config.VERBOSE_AGENT

        self._llm = ChatOllama(
            model=self._model,
            base_url=config.OLLAMA_BASE_URL,
            temperature=self._temperature,
            num_predict=4096,
        )

        self._tools = [
            get_portfolio_tool,
            get_news_feed_tool,
            get_market_data_tool,
        ]

        self._agent: CompiledStateGraph = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=SYSTEM_PROMPT,
        )

        self._chat_history: list[HumanMessage | AIMessage] = []

    def run_full_analysis(self, additional_instructions: str = "") -> str:
        """Run a comprehensive portfolio analysis.

        Instructs the agent to analyze portfolio, fetch news, get market data,
        and provide recommendations.

        Args:
            additional_instructions: Optional extra instructions for the analysis.

        Returns:
            Complete analysis report as a string.
        """
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(additional_instructions=additional_instructions)
        messages = list(self._chat_history) + [HumanMessage(content=prompt)]
        result = self._agent.invoke({"messages": messages})
        return _extract_response(result)

    def query(self, question: str) -> str:
        """Ask a specific investment question.

        Args:
            question: Natural language investment question.

        Returns:
            Agent's data-driven response.
        """
        prompt = QUERY_PROMPT_TEMPLATE.format(query=question)
        messages = list(self._chat_history) + [HumanMessage(content=prompt)]
        result = self._agent.invoke({"messages": messages})
        return _extract_response(result)

    def chat(self, message: str) -> str:
        """Interactive chat with the agent, maintaining conversation history.

        Args:
            message: User message.

        Returns:
            Agent's response.
        """
        self._chat_history.append(HumanMessage(content=message))
        result = self._agent.invoke({"messages": list(self._chat_history)})
        response = _extract_response(result)
        self._chat_history.append(AIMessage(content=response))
        return response

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._chat_history.clear()
