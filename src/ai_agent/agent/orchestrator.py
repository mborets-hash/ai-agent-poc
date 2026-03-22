"""LangChain-based agent orchestrator for investment analysis using local Ollama."""

from __future__ import annotations

import logging

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from ai_agent import config
from ai_agent.agent.prompts import (
    ANALYSIS_PROMPT_TEMPLATE,
    QUERY_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
)
from ai_agent.tools import get_market_data_tool, get_news_feed_tool, get_portfolio_tool

logger = logging.getLogger(__name__)


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

        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self._agent = create_tool_calling_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=self._prompt,
        )

        self._executor = AgentExecutor(
            agent=self._agent,
            tools=self._tools,
            verbose=self._verbose,
            max_iterations=config.MAX_AGENT_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

        self._chat_history: list[HumanMessage] = []

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

        result = self._executor.invoke(
            {
                "input": prompt,
                "chat_history": self._chat_history,
            }
        )

        return str(result["output"])

    def query(self, question: str) -> str:
        """Ask a specific investment question.

        Args:
            question: Natural language investment question.

        Returns:
            Agent's data-driven response.
        """
        prompt = QUERY_PROMPT_TEMPLATE.format(query=question)

        result = self._executor.invoke(
            {
                "input": prompt,
                "chat_history": self._chat_history,
            }
        )

        return str(result["output"])

    def chat(self, message: str) -> str:
        """Interactive chat with the agent, maintaining conversation history.

        Args:
            message: User message.

        Returns:
            Agent's response.
        """
        result = self._executor.invoke(
            {
                "input": message,
                "chat_history": self._chat_history,
            }
        )

        self._chat_history.append(HumanMessage(content=message))

        return str(result["output"])

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._chat_history.clear()
