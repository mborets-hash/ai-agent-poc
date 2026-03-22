"""LlamaIndex-based document store for indexing and retrieving financial documents."""

from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from ai_agent import config
from ai_agent.models.schemas import NewsArticle

logger = logging.getLogger(__name__)


class DocumentStore:
    """Manages document indexing and retrieval using LlamaIndex.

    Uses a vector store index to enable semantic search over financial
    news articles and research documents. Connects to a local Ollama
    server for LLM inference and embeddings.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
    ) -> None:
        self._persist_dir = Path(persist_dir) if persist_dir else config.INDEX_STORAGE_DIR
        self._index: VectorStoreIndex | None = None

        Settings.llm = Ollama(
            model=llm_model or config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=config.OLLAMA_TEMPERATURE,
            request_timeout=config.OLLAMA_REQUEST_TIMEOUT,
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=embedding_model or config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    def _load_or_create_index(self) -> VectorStoreIndex:
        """Load existing index from storage or create a new empty one."""
        if self._index is not None:
            return self._index

        if self._persist_dir.exists() and (self._persist_dir / "docstore.json").exists():
            logger.info("Loading existing index from %s", self._persist_dir)
            storage_context = StorageContext.from_defaults(persist_dir=str(self._persist_dir))
            self._index = load_index_from_storage(storage_context)
        else:
            logger.info("Creating new empty index")
            self._index = VectorStoreIndex(nodes=[])

        return self._index

    def index_news_articles(self, articles: list[NewsArticle]) -> int:
        """Index a list of news articles into the vector store.

        Args:
            articles: List of NewsArticle objects to index.

        Returns:
            Number of articles successfully indexed.
        """
        documents: list[Document] = []

        for article in articles:
            content = f"Title: {article.title}\n\n{article.summary}"
            metadata = {
                "source": article.source,
                "url": article.url,
                "title": article.title,
            }
            if article.published:
                metadata["published"] = article.published.isoformat()
            if article.related_symbols:
                metadata["symbols"] = ", ".join(article.related_symbols)

            documents.append(Document(text=content, metadata=metadata))

        if not documents:
            logger.warning("No documents to index")
            return 0

        index = self._load_or_create_index()
        for doc in documents:
            index.insert(doc)

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(self._persist_dir))
        logger.info("Indexed %d articles", len(documents))

        return len(documents)

    def query(self, query_text: str, top_k: int = 5) -> str:
        """Query the document store for relevant information.

        Args:
            query_text: Natural language query.
            top_k: Number of top results to return.

        Returns:
            Response text from the query engine.
        """
        index = self._load_or_create_index()
        query_engine = index.as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query_text)
        return str(response)

    def retrieve(self, query_text: str, top_k: int = 5) -> list[dict[str, str]]:
        """Retrieve relevant documents without LLM synthesis.

        Args:
            query_text: Natural language query.
            top_k: Number of top results to return.

        Returns:
            List of dicts with 'text', 'score', and metadata for each result.
        """
        index = self._load_or_create_index()
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query_text)

        results: list[dict[str, str]] = []
        for node in nodes:
            result = {
                "text": node.get_text(),
                "score": str(node.get_score()),
            }
            if node.metadata:
                for key, value in node.metadata.items():
                    result[key] = str(value)
            results.append(result)

        return results

    def clear(self) -> None:
        """Clear the index and remove persisted storage."""
        self._index = None
        if self._persist_dir.exists():
            import shutil

            shutil.rmtree(self._persist_dir)
            logger.info("Cleared index storage at %s", self._persist_dir)
