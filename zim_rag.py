#!/usr/bin/env python3
"""
ZIM to Vector Database RAG System

Turns a library of ZIM files (offline Wikipedia, Stack Exchange, docs, ...) into a
local vector database and answers questions about it with a local or hosted LLM.

Pipeline: ZIM (libzim) -> HTML cleaning -> chunking -> sentence-transformers
embeddings -> ChromaDB or FAISS -> retrieval -> LLM answer with numbered sources.

Usage:
    python zim_rag.py build
    python zim_rag.py query "What are the latest treatments for PTSD?"
    python zim_rag.py rag-query "Where was Ray Charles born?"
    python zim_rag.py chat
    python zim_rag.py info
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import click
import numpy as np
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from libzim.reader import Archive
from tqdm import tqdm

logger = logging.getLogger("zim_rag")

VECTOR_DB_TYPES = ("chroma", "faiss")
LLM_PROVIDERS = ("docker_model_runner", "ollama", "openai", "anthropic")

# Defaults applied when llm_model / llm_base_url are not set in the config.
DEFAULT_LLM_MODELS = {
    "docker_model_runner": "ai/smollm3:Q4_K_M",
    "ollama": "llama3.2",
    "anthropic": "claude-opus-5-5",
}
DEFAULT_LLM_BASE_URLS = {
    "docker_model_runner": "http://localhost:12434/engines/v1",
    "ollama": "http://localhost:11434/v1",
}

MANIFEST_VERSION = 1


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """Configuration for the ZIM RAG system."""

    zim_library_path: str = "./zim_library"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Instruction prefixes some retrieval models expect; auto-detected when None.
    embedding_query_prefix: Optional[str] = None
    embedding_document_prefix: Optional[str] = None
    embedding_device: Optional[str] = None  # e.g. 'cpu', 'cuda', 'mps'; auto when None
    embedding_batch_size: int = 64
    vector_db_type: str = "chroma"  # 'chroma' or 'faiss'
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_article_length: int = 200  # characters of clean text; shorter pages are skipped
    persist_directory: str = "./vector_db"
    collection_name: str = "zim_articles"
    llm_provider: str = "docker_model_runner"
    llm_model: Optional[str] = None  # provider default when None
    llm_base_url: Optional[str] = None  # provider default when None
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    llm_timeout: float = 300.0
    retrieval_k: int = 5
    max_context_chars: int = 8000
    max_articles_per_zim: Optional[int] = None

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        """Load config from ``path``, or from ./config.json when it exists."""
        if path is None:
            if not Path("config.json").is_file():
                return cls()
            path = "config.json"

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise click.ClickException(f"Config file not found: {path}")
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in {path}: {e}")

        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise click.ClickException(
                f"Unknown config option(s) in {path}: {', '.join(unknown)}. "
                f"Valid options: {', '.join(sorted(known))}"
            )
        config = cls(**data)
        config.validate()
        return config

    def validate(self):
        if self.vector_db_type not in VECTOR_DB_TYPES:
            raise click.ClickException(
                f"vector_db_type must be one of {VECTOR_DB_TYPES}, got '{self.vector_db_type}'"
            )
        if self.llm_provider not in LLM_PROVIDERS:
            raise click.ClickException(
                f"llm_provider must be one of {LLM_PROVIDERS}, got '{self.llm_provider}'"
            )
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise click.ClickException("chunk_overlap must be >= 0 and < chunk_size")

    @property
    def resolved_llm_model(self) -> str:
        model = self.llm_model or DEFAULT_LLM_MODELS.get(self.llm_provider)
        if not model:
            raise click.ClickException(
                f"Set 'llm_model' in the config for provider '{self.llm_provider}'"
            )
        return model


# --------------------------------------------------------------------------- #
# ZIM reading
# --------------------------------------------------------------------------- #


@dataclass
class Article:
    """An HTML article extracted from a ZIM file."""

    url: str
    title: str
    content: str
    source_zim: str = ""
    metadata: Dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable ID, unique across ZIM files (the same path can exist in many ZIMs)."""
        return hashlib.sha1(f"{self.source_zim}/{self.url}".encode()).hexdigest()

    def to_dict(self) -> Dict:
        return asdict(self)


class ZIMReader:
    """Reads articles and metadata from a ZIM file using libzim."""

    def __init__(self, zim_path):
        self.path = Path(zim_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"ZIM file not found: {zim_path}")
        self.archive = Archive(str(self.path))

    @property
    def uuid(self) -> str:
        return str(self.archive.uuid)

    def _metadata(self, key: str) -> str:
        try:
            return bytes(self.archive.get_metadata(key)).decode("utf-8", "replace").strip()
        except (KeyError, RuntimeError):
            return ""

    def get_metadata(self) -> Dict:
        return {
            "filename": self.path.name,
            "title": self._metadata("Title") or self.path.stem,
            "description": self._metadata("Description"),
            "language": self._metadata("Language"),
            "creator": self._metadata("Creator"),
            "date": self._metadata("Date"),
            "article_count": self.archive.article_count,
            "size_mb": round(self.path.stat().st_size / (1024 * 1024), 1),
            "uuid": self.uuid,
        }

    def iter_articles(
        self, start: int = 0, progress: bool = True
    ) -> Iterator[Tuple[int, Article]]:
        """Yield ``(entry_index, article)`` for every HTML article from ``start`` on.

        Redirects and non-HTML entries (CSS, JS, images, metadata) are skipped.
        The entry index lets callers checkpoint and resume long extractions.
        """
        total = self.archive.all_entry_count
        bar = tqdm(
            total=total,
            initial=start,
            desc=self.path.name,
            unit="entry",
            disable=not progress,
            leave=False,
        )
        try:
            for i in range(start, total):
                bar.update(1)
                try:
                    # python-libzim has no public API to iterate all entries.
                    entry = self.archive._get_entry_by_id(i)
                    if entry.is_redirect:
                        continue
                    item = entry.get_item()
                    if not item.mimetype.startswith("text/html"):
                        continue
                    content = bytes(item.content).decode("utf-8", errors="replace")
                except Exception as e:
                    logger.debug(f"Skipping entry {i} of {self.path.name}: {e}")
                    continue

                yield i, Article(
                    url=item.path,
                    title=item.title or item.path,
                    content=content,
                    source_zim=self.path.name,
                )
        finally:
            bar.close()


class ZIMLibraryManager:
    """Manages a collection of ZIM files in a library directory."""

    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        if not self.library_path.exists():
            logger.info(f"Creating ZIM library directory: {self.library_path}")
            self.library_path.mkdir(parents=True, exist_ok=True)
        self.zim_files = sorted(self.library_path.glob("*.zim"))

    def resolve(self, zim_file: str) -> Path:
        """Resolve a ZIM file name (in the library) or path."""
        candidate = Path(zim_file)
        if candidate.is_file():
            return candidate
        candidate = self.library_path / zim_file
        if candidate.is_file():
            return candidate
        raise click.ClickException(
            f"ZIM file not found: {zim_file} (looked in {self.library_path})"
        )

    def select(self, zim_files: Iterable[str] = ()) -> List[Path]:
        zim_files = list(zim_files)
        if zim_files:
            return [self.resolve(z) for z in zim_files]
        if not self.zim_files:
            raise click.ClickException(
                f"No ZIM files found in library: {self.library_path}. "
                "Download some from https://library.kiwix.org/"
            )
        return self.zim_files

    def get_library_info(self) -> Dict:
        info = {
            "library_path": str(self.library_path),
            "total_zim_files": len(self.zim_files),
            "zim_files": [],
        }
        for zim_path in self.zim_files:
            try:
                info["zim_files"].append(ZIMReader(zim_path).get_metadata())
            except Exception as e:
                logger.warning(f"Could not read metadata for {zim_path.name}: {e}")
                info["zim_files"].append(
                    {"filename": zim_path.name, "title": zim_path.stem, "error": str(e)}
                )
        return info


# --------------------------------------------------------------------------- #
# Text processing
# --------------------------------------------------------------------------- #

# Page furniture that adds noise to embeddings (mostly MediaWiki/mwoffliner markup).
NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "template",
    "head",
    "nav",
    "footer",
    "sup.reference",
    "sup.mw-ref",
    "ol.references",
    ".mw-references-wrap",
    ".reflist",
    ".navbox",
    ".navbox-styles",
    ".vertical-navbox",
    ".mw-editsection",
    ".mw-jump-link",
    ".noprint",
    ".mw-empty-elt",
    ".metadata",
    ".sistersitebox",
    ".hatnote",
    ".catlinks",
    ".printfooter",
    ".zim-footer",
    "#toc",
    ".toc",
]

BLOCK_TAGS = [
    "address", "article", "aside", "blockquote", "br", "caption", "dd", "details",
    "div", "dl", "dt", "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6",
    "header", "hr", "li", "main", "ol", "p", "pre", "section", "summary", "table",
    "ul",
]


_BLOCK, _PRE_NEWLINE, _PRE_SPACE = "\x00", "\x01", "\x02"


def _html_parser() -> str:
    try:
        import lxml  # noqa: F401

        return "lxml"
    except ImportError:
        return "html.parser"


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict


class TextProcessor:
    """Cleans article HTML and splits it into chunks for embedding."""

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200, min_length: int = 200
    ):
        self.min_length = min_length
        self.parser = _html_parser()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator="end",
        )

    def clean_html(self, html_content: str) -> str:
        """Extract readable text, keeping paragraph boundaries and table rows."""
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, self.parser)
        root = soup.body or soup

        for element in root.select(",".join(NOISE_SELECTORS)):
            element.decompose()

        # Whitespace in HTML source is insignificant except inside <pre>, so
        # protect preformatted text and mark block boundaries with sentinels.
        for pre in root.find_all("pre"):
            for string in pre.find_all(string=True):
                string.replace_with(string.replace("\n", _PRE_NEWLINE).replace(" ", _PRE_SPACE))

        # Flatten table rows to "cell | cell" lines so infoboxes stay readable.
        for row in root.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            row.replace_with(_BLOCK + " | ".join(c for c in cells if c) + _BLOCK)

        for element in root.find_all(BLOCK_TAGS):
            element.insert_before(_BLOCK)
            element.insert_after(_BLOCK)

        paragraphs = []
        for part in root.get_text().split(_BLOCK):
            part = re.sub(r"\s+", " ", part).strip()
            if part:
                paragraphs.append(part.replace(_PRE_NEWLINE, "\n").replace(_PRE_SPACE, " "))
        return "\n\n".join(paragraphs)

    def chunk_article(self, article: Article) -> List[Chunk]:
        text = self.clean_html(article.content)
        if len(text) < self.min_length:
            return []

        pieces = self.text_splitter.split_text(text)
        return [
            Chunk(
                id=f"{article.id}_{i}",
                text=piece,
                metadata={
                    "title": article.title,
                    "url": article.url,
                    "source_zim": article.source_zim,
                    "article_id": article.id,
                    "chunk_index": i,
                    "total_chunks": len(pieces),
                },
            )
            for i, piece in enumerate(pieces)
        ]

    @staticmethod
    def embedding_text(chunk: Chunk) -> str:
        """Text that gets embedded: the title gives each chunk its topic."""
        return f"{chunk.metadata['title']}\n\n{chunk.text}"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #


# Prefixes that instruction-tuned retrieval models were trained with.
KNOWN_EMBEDDING_PREFIXES = [
    # (pattern, query prefix, document prefix)
    (r"bge-(small|base|large)-en", "Represent this sentence for searching relevant passages: ", ""),
    (r"(^|/)(multilingual-)?e5-", "query: ", "passage: "),
]


def embedding_prefixes(config: Config, model_name: str) -> Tuple[str, str]:
    """Return (query_prefix, document_prefix) for a model, honouring config overrides."""
    query_prefix, document_prefix = "", ""
    for pattern, query, document in KNOWN_EMBEDDING_PREFIXES:
        if re.search(pattern, model_name, re.IGNORECASE):
            query_prefix, document_prefix = query, document
            break
    if config.embedding_query_prefix is not None:
        query_prefix = config.embedding_query_prefix
    if config.embedding_document_prefix is not None:
        document_prefix = config.embedding_document_prefix
    return query_prefix, document_prefix


class Embedder:
    """Sentence-transformers embeddings, L2-normalised for cosine similarity."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 64,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        try:
            # faiss and torch bundle separate OpenMP runtimes; on macOS loading
            # torch first makes FAISS searches segfault, so load faiss first.
            import faiss  # noqa: F401
        except ImportError:
            pass
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        logger.info(f"Loading embedding model: {model_name}")
        try:
            # Use the local cache first so queries work offline.
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            logger.info("Embedding model not cached locally, downloading it")
            self.model = SentenceTransformer(model_name, device=device)

    @property
    def dimension(self) -> int:
        get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension"
        )
        return int(get_dim())

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([self.query_prefix + text])[0]


def load_embedder(
    config: Config, model_name: Optional[str] = None, prefixes: Optional[Tuple[str, str]] = None
) -> Embedder:
    model_name = model_name or config.embedding_model
    query_prefix, document_prefix = prefixes or embedding_prefixes(config, model_name)
    return Embedder(
        model_name,
        device=config.embedding_device,
        batch_size=config.embedding_batch_size,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )


# --------------------------------------------------------------------------- #
# Vector stores
# --------------------------------------------------------------------------- #


@dataclass
class SearchResult:
    text: str
    metadata: Dict
    score: float  # cosine similarity, higher is better

    @property
    def title(self) -> str:
        return self.metadata.get("title", "Unknown")

    @property
    def url(self) -> str:
        return self.metadata.get("url", "")

    @property
    def source_zim(self) -> str:
        return self.metadata.get("source_zim", "")


class ChromaStore:
    """ChromaDB-backed store using cosine distance."""

    def __init__(self, persist_directory: str, collection_name: str):
        import chromadb
        from chromadb.config import Settings

        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self._open_collection()
        try:
            self.max_batch = min(self.client.get_max_batch_size(), 1000)
        except Exception:
            self.max_batch = 500

    def _open_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
            metadata={"description": "ZIM articles for RAG"},
        )

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        for i in range(0, len(chunks), self.max_batch):
            batch = chunks[i : i + self.max_batch]
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embeddings[i : i + self.max_batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )

    def search(
        self, embedding: np.ndarray, k: int, source_zim: Optional[str] = None
    ) -> List[SearchResult]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where={"source_zim": source_zim} if source_zim else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            SearchResult(text=doc, metadata=meta or {}, score=1.0 - dist)
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def delete_source(self, source_zim: str):
        self.collection.delete(where={"source_zim": source_zim})

    def count(self) -> int:
        return self.collection.count()

    def persist(self):
        pass  # ChromaDB persists every write

    def reset(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._open_collection()


class FaissStore:
    """FAISS inner-product index plus a SQLite table holding chunk text and metadata.

    Vector IDs are the SQLite row IDs, so chunks of a ZIM file can be removed
    from both the index and the table.
    """

    def __init__(self, persist_directory: str, collection_name: str):
        import faiss

        self.faiss = faiss
        self.directory = Path(persist_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_path = self.directory / f"{collection_name}.faiss"
        self.db_path = self.directory / f"{collection_name}.sqlite"
        self.db = sqlite3.connect(self.db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "id INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, "
            "source_zim TEXT NOT NULL, text TEXT NOT NULL, metadata TEXT NOT NULL)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source_zim)")
        self.index = faiss.read_index(str(self.index_path)) if self.index_path.exists() else None

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        if self.index is None:
            self.index = self.faiss.IndexIDMap2(self.faiss.IndexFlatIP(embeddings.shape[1]))

        # Replace chunks that were already stored (e.g. after an interrupted build).
        existing = []
        for i in range(0, len(chunks), 500):
            ids = [c.id for c in chunks[i : i + 500]]
            existing += self.db.execute(
                f"SELECT id FROM chunks WHERE chunk_id IN ({','.join('?' * len(ids))})", ids
            ).fetchall()
        if existing:
            self._remove_rows([row[0] for row in existing])

        row_ids = []
        for chunk in chunks:
            cursor = self.db.execute(
                "INSERT INTO chunks (chunk_id, source_zim, text, metadata) VALUES (?, ?, ?, ?)",
                (chunk.id, chunk.metadata["source_zim"], chunk.text, json.dumps(chunk.metadata)),
            )
            row_ids.append(cursor.lastrowid)
        self.index.add_with_ids(
            np.ascontiguousarray(embeddings, dtype=np.float32),
            np.asarray(row_ids, dtype=np.int64),
        )

    def _remove_rows(self, row_ids: List[int]):
        if self.index is not None:
            self.index.remove_ids(np.asarray(row_ids, dtype=np.int64))
        self.db.executemany("DELETE FROM chunks WHERE id = ?", [(r,) for r in row_ids])

    def search(
        self, embedding: np.ndarray, k: int, source_zim: Optional[str] = None
    ) -> List[SearchResult]:
        if self.index is None or self.index.ntotal == 0:
            return []

        params = None
        if source_zim:
            ids = [
                r[0]
                for r in self.db.execute(
                    "SELECT id FROM chunks WHERE source_zim = ?", (source_zim,)
                )
            ]
            if not ids:
                return []
            params = self.faiss.SearchParameters(
                sel=self.faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))
            )

        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        scores, ids = self.index.search(query, k, params=params)

        results = []
        for score, row_id in zip(scores[0], ids[0]):
            if row_id < 0:
                continue
            row = self.db.execute(
                "SELECT text, metadata FROM chunks WHERE id = ?", (int(row_id),)
            ).fetchone()
            if row:
                results.append(SearchResult(row[0], json.loads(row[1]), float(score)))
        return results

    def delete_source(self, source_zim: str):
        rows = self.db.execute(
            "SELECT id FROM chunks WHERE source_zim = ?", (source_zim,)
        ).fetchall()
        if rows:
            self._remove_rows([r[0] for r in rows])
        self.persist()

    def count(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def persist(self):
        """Write the index and commit chunk rows together, so both stay in sync."""
        if self.index is not None:
            tmp_path = self.index_path.with_suffix(".faiss.tmp")
            self.faiss.write_index(self.index, str(tmp_path))
            os.replace(tmp_path, self.index_path)
        self.db.commit()

    def reset(self):
        self.db.execute("DELETE FROM chunks")
        self.db.commit()
        self.index = None
        self.index_path.unlink(missing_ok=True)


def open_store(config: Config, vector_db_type: Optional[str] = None):
    kind = vector_db_type or config.vector_db_type
    Path(config.persist_directory).mkdir(parents=True, exist_ok=True)
    if kind == "chroma":
        return ChromaStore(config.persist_directory, config.collection_name)
    if kind == "faiss":
        return FaissStore(config.persist_directory, config.collection_name)
    raise click.ClickException(f"Unsupported vector database type: {kind}")


# --------------------------------------------------------------------------- #
# Manifest: which ZIM files are indexed, and with which settings
# --------------------------------------------------------------------------- #


class Manifest:
    """JSON file next to the vector store that records how it was built."""

    def __init__(self, path: Path, data: Optional[Dict] = None):
        self.path = path
        self.data = data or {"version": MANIFEST_VERSION, "zims": {}}

    @classmethod
    def load(cls, config: Config) -> "Manifest":
        path = Path(config.persist_directory) / f"{config.collection_name}.manifest.json"
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return cls(path, json.load(f))
        return cls(path)

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    @property
    def zims(self) -> Dict[str, Dict]:
        return self.data.setdefault("zims", {})

    @property
    def embedding_model(self) -> Optional[str]:
        return self.data.get("embedding_model")

    @property
    def vector_db_type(self) -> Optional[str]:
        return self.data.get("vector_db_type")

    def total_chunks(self) -> int:
        return sum(z.get("chunks", 0) for z in self.zims.values())

    @property
    def embedding_prefixes(self) -> Optional[Tuple[str, str]]:
        prefixes = self.data.get("embedding_prefixes")
        return (prefixes["query"], prefixes["document"]) if prefixes else None

    def initialise(self, config: Config, embedding_model: str):
        query_prefix, document_prefix = embedding_prefixes(config, embedding_model)
        self.data = {
            "version": MANIFEST_VERSION,
            "vector_db_type": config.vector_db_type,
            "embedding_model": embedding_model,
            "embedding_prefixes": {"query": query_prefix, "document": document_prefix},
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "zims": {},
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp_path, self.path)


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


class Indexer:
    """Streams articles from ZIM files into the vector store.

    Memory use stays flat regardless of ZIM size, and progress is checkpointed
    to the manifest so an interrupted build resumes where it stopped.
    """

    flush_chunks = 512  # chunks embedded and stored per batch
    checkpoint_seconds = 60.0

    def __init__(self, config: Config, store, embedder: Embedder, manifest: Manifest):
        self.config = config
        self.store = store
        self.embedder = embedder
        self.manifest = manifest
        self.processor = TextProcessor(
            config.chunk_size, config.chunk_overlap, config.min_article_length
        )

    def index_zim(self, zim_path: Path, limit: Optional[int] = None) -> bool:
        """Index one ZIM file. Returns False when it was already up to date."""
        name = zim_path.name
        reader = ZIMReader(zim_path)
        previous = self.manifest.zims.get(name)
        same_input = (
            previous is not None
            and previous.get("uuid") == reader.uuid
            and previous.get("limit") == limit
        )

        if same_input and previous.get("status") == "complete":
            logger.info(f"{name}: already indexed ({previous.get('chunks', 0)} chunks), skipping")
            return False

        if same_input and previous.get("status") == "in_progress":
            state = previous
            logger.info(
                f"{name}: resuming at entry {state['next_entry']} "
                f"({state['articles']} articles already indexed)"
            )
        else:
            if previous is not None:
                logger.info(f"{name}: ZIM file or options changed, re-indexing")
                self.store.delete_source(name)
            state = {
                "uuid": reader.uuid,
                "limit": limit,
                "status": "in_progress",
                "next_entry": 0,
                "articles": 0,
                "chunks": 0,
            }
            self.manifest.zims[name] = state
            self.manifest.save()
            logger.info(f"{name}: indexing")

        pending: List[Chunk] = []
        pending_articles = 0
        last_checkpoint = time.monotonic()

        def flush(next_entry: int):
            nonlocal pending, pending_articles, last_checkpoint
            if pending:
                texts = [TextProcessor.embedding_text(c) for c in pending]
                self.store.add(pending, self.embedder.embed_documents(texts))
            state["articles"] += pending_articles
            state["chunks"] += len(pending)
            state["next_entry"] = next_entry
            pending, pending_articles = [], 0
            if time.monotonic() - last_checkpoint >= self.checkpoint_seconds:
                self.store.persist()
                self.manifest.save()
                last_checkpoint = time.monotonic()

        next_entry = state["next_entry"]
        try:
            for entry_index, article in reader.iter_articles(start=state["next_entry"]):
                next_entry = entry_index + 1
                if limit is not None and state["articles"] + pending_articles >= limit:
                    break
                chunks = self.processor.chunk_article(article)
                if not chunks:
                    continue
                pending.extend(chunks)
                pending_articles += 1
                if len(pending) >= self.flush_chunks:
                    flush(next_entry)
            else:
                next_entry = reader.archive.all_entry_count
            flush(next_entry)
        except KeyboardInterrupt:
            # Everything up to the last flush is stored; save it so a rerun resumes.
            self.store.persist()
            self.manifest.save()
            raise

        state["status"] = "complete"
        state["indexed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.store.persist()
        self.manifest.save()
        logger.info(f"{name}: indexed {state['articles']} articles as {state['chunks']} chunks")
        return True


# --------------------------------------------------------------------------- #
# Retrieval and generation
# --------------------------------------------------------------------------- #


class KnowledgeBase:
    """Read-side access to a built vector database."""

    def __init__(self, config: Config, embedder: Optional[Embedder] = None):
        self.config = config
        self.manifest = Manifest.load(config)
        if not self.manifest.exists or not self.manifest.zims:
            raise click.ClickException(
                f"Vector database not found in {config.persist_directory}. "
                "Run 'python zim_rag.py build' first."
            )
        if self.manifest.vector_db_type != config.vector_db_type:
            raise click.ClickException(
                f"The database in {config.persist_directory} was built with "
                f"'{self.manifest.vector_db_type}', but the config says "
                f"'{config.vector_db_type}'."
            )

        model = self.manifest.embedding_model
        if model != config.embedding_model:
            logger.info(f"Using embedding model '{model}' that the database was built with")
        self.embedder = embedder or load_embedder(config, model, self.manifest.embedding_prefixes)
        self.store = open_store(config)

    def search(
        self, question: str, k: Optional[int] = None, source_zim: Optional[str] = None
    ) -> List[SearchResult]:
        embedding = self.embedder.embed_query(question)
        return self.store.search(embedding, k or self.config.retrieval_k, source_zim)


SYSTEM_PROMPT = (
    "You answer questions using only the numbered excerpts from an offline "
    "knowledge base that are given to you. Cite the excerpts you use with their "
    "numbers, like [1] or [2][3]. If the excerpts do not contain the answer, say "
    "that you don't know instead of guessing. Answer concisely."
)


def build_prompt(question: str, results: List[SearchResult], max_chars: int) -> str:
    """Format retrieved excerpts as numbered context, within a character budget."""
    parts, used = [], 0
    for i, result in enumerate(results, 1):
        header = f"[{i}] {result.title} ({result.source_zim})"
        remaining = max_chars - used - len(header)
        if remaining < 200:
            break
        text = result.text if len(result.text) <= remaining else result.text[:remaining] + "..."
        parts.append(f"{header}\n{text}")
        used += len(header) + len(text)
    context = "\n\n".join(parts) if parts else "(no excerpts found)"
    return f"Excerpts:\n\n{context}\n\nQuestion: {question}"


def strip_think_tags(pieces: Iterable[str]) -> Iterator[str]:
    """Drop <think>...</think> reasoning that some local models (SmolLM3, Qwen3) emit."""
    open_tag, close_tag = "<think>", "</think>"
    buffer, inside, at_start = "", False, True

    def partial_tag_len(text: str, tag: str) -> int:
        for n in range(min(len(tag) - 1, len(text)), 0, -1):
            if text.endswith(tag[:n]):
                return n
        return 0

    for piece in pieces:
        buffer += piece
        while buffer:
            if inside:
                end = buffer.find(close_tag)
                if end == -1:
                    buffer = buffer[-(len(close_tag) - 1) :]
                    break
                buffer = buffer[end + len(close_tag) :]
                inside, at_start = False, True
                continue

            start = buffer.find(open_tag)
            if start == -1:
                keep = partial_tag_len(buffer, open_tag)
                out, buffer = buffer[: len(buffer) - keep], buffer[len(buffer) - keep :]
            else:
                out, buffer, inside = buffer[:start], buffer[start + len(open_tag) :], True
            if at_start:
                out = out.lstrip()
                at_start = not out
            if out:
                yield out
            if start == -1:
                break

    if not inside and buffer:
        yield buffer.lstrip() if at_start else buffer


class LLMError(click.ClickException):
    pass


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint: Docker Model Runner, Ollama, OpenAI,
    LM Studio, llama.cpp server, vLLM, ..."""

    def __init__(self, config: Config):
        import openai

        self.openai = openai
        self.config = config
        self.model = config.resolved_llm_model
        base_url = config.llm_base_url or DEFAULT_LLM_BASE_URLS.get(config.llm_provider)
        api_key = os.environ.get("OPENAI_API_KEY")
        if config.llm_provider != "openai":
            api_key = api_key or "not-needed"  # local servers ignore the key
        elif not api_key:
            raise LLMError("Set the OPENAI_API_KEY environment variable to use provider 'openai'")
        self.base_url = base_url
        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=config.llm_timeout, max_retries=1
        )

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        kwargs = {}
        if self.config.llm_temperature is not None:
            kwargs["temperature"] = self.config.llm_temperature
        if self.config.llm_max_tokens is not None:
            kwargs["max_tokens"] = self.config.llm_max_tokens
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                **kwargs,
            )
            for event in response:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except self.openai.APITimeoutError:
            raise LLMError(
                f"The LLM did not answer within {self.config.llm_timeout:.0f}s "
                "(raise llm_timeout in the config for slow machines)"
            )
        except self.openai.APIConnectionError as e:
            if self.config.llm_provider == "docker_model_runner":
                logger.info(
                    "Docker Model Runner API not reachable; falling back to the docker CLI. "
                    "Enable it with: docker desktop enable model-runner --tcp 12434"
                )
                yield from docker_cli_generate(self.model, system, prompt, self.config.llm_timeout)
                return
            raise LLMError(f"Could not reach the LLM server at {self.base_url}: {e}")
        except self.openai.NotFoundError:
            hint = ""
            if self.config.llm_provider == "docker_model_runner":
                hint = f" Pull it with: docker model pull {self.model}"
            elif self.config.llm_provider == "ollama":
                hint = f" Pull it with: ollama pull {self.model}"
            raise LLMError(f"Model '{self.model}' not found.{hint}")
        except self.openai.APIStatusError as e:
            raise LLMError(f"LLM request failed ({e.status_code}): {e.message}")


def docker_cli_generate(model: str, system: str, prompt: str, timeout: float) -> Iterator[str]:
    """One-shot generation through ``docker model run`` (no streaming)."""
    try:
        result = subprocess.run(
            ["docker", "model", "run", model, f"{system}\n\n{prompt}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise LLMError("Docker not found. Install Docker Desktop with Model Runner enabled.")
    except subprocess.TimeoutExpired:
        raise LLMError(f"docker model run timed out after {timeout:.0f}s")
    if result.returncode != 0:
        raise LLMError(f"docker model run failed: {result.stderr.strip()}")
    yield result.stdout.strip()


class AnthropicLLM:
    """Claude via the Anthropic Messages API (reads ANTHROPIC_API_KEY)."""

    def __init__(self, config: Config):
        import anthropic

        self.anthropic = anthropic
        self.config = config
        self.model = config.resolved_llm_model
        kwargs = {"timeout": config.llm_timeout}
        if config.llm_base_url:
            kwargs["base_url"] = config.llm_base_url
        self.client = anthropic.Anthropic(**kwargs)

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.config.llm_max_tokens or 16000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                # Retry declined requests on a fallback model inside the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                yield from stream.text_stream
                message = stream.get_final_message()
            if message.stop_reason == "refusal":
                yield "\n[The model declined to answer this request.]"
        except self.anthropic.AuthenticationError:
            raise LLMError("Anthropic authentication failed. Set ANTHROPIC_API_KEY.")
        except self.anthropic.APIConnectionError as e:
            raise LLMError(f"Could not reach the Anthropic API: {e}")
        except self.anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic request failed ({e.status_code}): {e.message}")


def make_llm(config: Config):
    if config.llm_provider == "anthropic":
        return AnthropicLLM(config)
    return OpenAICompatibleLLM(config)


class RAGSystem:
    """Retrieval-augmented generation over the knowledge base."""

    def __init__(self, config: Config, knowledge_base: Optional[KnowledgeBase] = None, llm=None):
        self.config = config
        self.kb = knowledge_base or KnowledgeBase(config)
        self.llm = llm or make_llm(config)

    def ask(
        self, question: str, k: Optional[int] = None, source_zim: Optional[str] = None
    ) -> Tuple[Iterator[str], List[SearchResult]]:
        """Return a stream of answer text and the excerpts it was based on."""
        results = self.kb.search(question, k, source_zim)
        prompt = build_prompt(question, results, self.config.max_context_chars)
        return strip_think_tags(self.llm.stream(SYSTEM_PROMPT, prompt)), results

    def query(self, question: str, k: Optional[int] = None, source_zim: Optional[str] = None) -> Dict:
        stream, results = self.ask(question, k, source_zim)
        return {
            "answer": "".join(stream).strip(),
            "sources": [{"score": r.score, **r.metadata} for r in results],
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    console.setLevel(level)
    logger.addHandler(console)

    try:
        log_file = logging.FileHandler("zim_rag.log", encoding="utf-8")
        log_file.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(log_file)
    except OSError:
        pass

    if not verbose:
        try:
            from transformers.utils import logging as hf_logging

            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
        except ImportError:
            pass


def print_results(results: List[SearchResult], snippet_chars: int = 300):
    for i, result in enumerate(results, 1):
        text = " ".join(result.text.split())
        if len(text) > snippet_chars:
            text = text[:snippet_chars] + "..."
        click.echo(f"[{i}] {result.title}  (score {result.score:.3f})")
        click.echo(f"    {result.source_zim} · {result.url}")
        click.echo(f"    {text}\n")


def stream_answer(rag: RAGSystem, question: str, k: Optional[int], source: Optional[str], show_sources: bool):
    stream, results = rag.ask(question, k, source)
    if not results:
        click.echo("No relevant excerpts found in the knowledge base.")
        return
    for piece in stream:
        click.echo(piece, nl=False)
    click.echo("\n")
    if show_sources:
        click.echo("Sources:")
        for i, r in enumerate(results, 1):
            click.echo(f"  [{i}] {r.title}  ({r.source_zim} · {r.url}, score {r.score:.3f})")


source_option = click.option(
    "--source", default=None, help="Only search chunks from this ZIM file name"
)
k_option = click.option(
    "--k", default=None, type=int, help="Number of chunks to retrieve [default: retrieval_k]"
)


@click.group()
@click.option(
    "--config", default=None, help="Path to config file [default: ./config.json if present]"
)
@click.option("-v", "--verbose", is_flag=True, help="Show debug logging")
@click.pass_context
def cli(ctx, config, verbose):
    """ZIM to Vector Database RAG System"""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.load(config)


@cli.command()
@click.option(
    "--zim-file",
    "zim_files",
    multiple=True,
    help="ZIM file (name in the library, or a path) to process; repeatable. Default: all",
)
@click.option("--limit", default=None, type=int, help="Limit number of articles per ZIM file")
@click.option("--force", is_flag=True, help="Delete the existing database and rebuild it")
@click.option("--embedding-model", default=None, help="Override the configured embedding model")
@click.pass_context
def build(ctx, zim_files, limit, force, embedding_model):
    """Build or update the vector database from the ZIM library.

    ZIM files that are already indexed are skipped, new ones are added, and an
    interrupted build resumes where it stopped.
    """
    config: Config = ctx.obj["config"]
    library = ZIMLibraryManager(config.zim_library_path)
    targets = library.select(zim_files)
    limit = limit if limit is not None else config.max_articles_per_zim
    model = embedding_model or config.embedding_model

    manifest = Manifest.load(config)
    store = open_store(config)

    if force:
        logger.info("Deleting existing vector database (--force)")
        store.reset()
        manifest.data = {}
    elif manifest.exists:
        if manifest.vector_db_type != config.vector_db_type:
            raise click.ClickException(
                f"Existing database uses '{manifest.vector_db_type}'; use --force to "
                f"rebuild it with '{config.vector_db_type}'."
            )
        if manifest.embedding_model != model:
            raise click.ClickException(
                f"Existing database was built with embedding model "
                f"'{manifest.embedding_model}'; use --force to rebuild it with '{model}'."
            )
        if (manifest.data.get("chunk_size"), manifest.data.get("chunk_overlap")) != (
            config.chunk_size,
            config.chunk_overlap,
        ):
            logger.warning(
                "chunk_size/chunk_overlap differ from the existing database; "
                "only newly indexed ZIM files will use the new values."
            )
    elif store.count() > 0:
        raise click.ClickException(
            f"{config.persist_directory} contains a database from an older version of "
            "this tool. Rebuild it with --force."
        )

    if not manifest.exists or force:
        manifest.initialise(config, model)
        manifest.save()

    embedder = load_embedder(config, model, manifest.embedding_prefixes)
    indexer = Indexer(config, store, embedder, manifest)

    started = time.monotonic()
    try:
        for zim_path in targets:
            indexer.index_zim(zim_path, limit)
    except KeyboardInterrupt:
        raise click.ClickException(
            "Interrupted. Progress was saved; run build again to resume."
        )

    missing = sorted(set(manifest.zims) - {p.name for p in library.zim_files})
    if missing and not zim_files:
        logger.warning(
            f"Indexed ZIM files no longer in the library: {', '.join(missing)}. "
            "Remove them with: python zim_rag.py remove <name>"
        )

    click.echo(
        f"Vector database ready: {len(manifest.zims)} ZIM file(s), "
        f"{manifest.total_chunks()} chunks ({time.monotonic() - started:.0f}s)."
    )


@cli.command()
@click.argument("question")
@k_option
@source_option
@click.option("--json", "as_json", is_flag=True, help="Print results as JSON")
@click.pass_context
def query(ctx, question, k, source, as_json):
    """Semantic search over the vector database (no LLM)."""
    kb = KnowledgeBase(ctx.obj["config"])
    results = kb.search(question, k, source)
    if as_json:
        click.echo(
            json.dumps(
                [{"score": r.score, "text": r.text, **r.metadata} for r in results],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    click.echo(f"\nQuery: {question}\nFound {len(results)} relevant chunks:\n")
    print_results(results)


@cli.command("rag-query")
@click.argument("question")
@k_option
@source_option
@click.option("--no-sources", is_flag=True, help="Do not list the source excerpts")
@click.pass_context
def rag_query(ctx, question, k, source, no_sources):
    """Answer a question with the LLM, grounded in retrieved excerpts."""
    rag = RAGSystem(ctx.obj["config"])
    click.echo(f"\nQuestion: {question}\n")
    stream_answer(rag, question, k, source, not no_sources)


@cli.command()
@k_option
@source_option
@click.option("--no-sources", is_flag=True, help="Do not list the source excerpts")
@click.pass_context
def chat(ctx, k, source, no_sources):
    """Interactive question answering (models stay loaded between questions)."""
    rag = RAGSystem(ctx.obj["config"])
    click.echo("Ask a question about your ZIM library. Type 'exit' or press Ctrl-D to quit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo()
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        try:
            stream_answer(rag, question, k, source, not no_sources)
        except LLMError as e:
            click.echo(f"Error: {e.message}", err=True)
        except KeyboardInterrupt:
            click.echo("\n(interrupted)")
        click.echo()


@cli.command()
@click.pass_context
def info(ctx):
    """Show information about the ZIM library and vector database."""
    config: Config = ctx.obj["config"]
    library_info = ZIMLibraryManager(config.zim_library_path).get_library_info()
    manifest = Manifest.load(config)

    click.echo("ZIM Library Information:")
    click.echo(f"  Library Path: {library_info['library_path']}")
    click.echo(f"  Total ZIM Files: {library_info['total_zim_files']}")

    for zim_info in library_info["zim_files"]:
        indexed = manifest.zims.get(zim_info["filename"], {})
        status = {
            "complete": f"✅ indexed ({indexed.get('chunks', 0)} chunks)",
            "in_progress": "⏸  partially indexed (run build to resume)",
        }.get(indexed.get("status"), "❌ not indexed")
        click.echo(f"\n  📁 {zim_info['filename']}")
        click.echo(f"    Title: {zim_info.get('title', 'Unknown')}")
        if "error" in zim_info:
            click.echo(f"    ⚠️  Error: {zim_info['error']}")
            continue
        click.echo(f"    Language: {zim_info.get('language') or 'Unknown'}")
        click.echo(f"    Creator: {zim_info.get('creator') or 'Unknown'}")
        click.echo(f"    Date: {zim_info.get('date') or 'Unknown'}")
        click.echo(f"    Articles: {zim_info.get('article_count')}")
        click.echo(f"    Size: {zim_info.get('size_mb')} MB")
        click.echo(f"    Status: {status}")

    click.echo("\nVector Database Information:")
    click.echo(f"  Status: {'✅ Built' if manifest.exists and manifest.zims else '❌ Not found'}")
    click.echo(f"  Type: {manifest.vector_db_type or config.vector_db_type}")
    click.echo(f"  Embedding Model: {manifest.embedding_model or config.embedding_model}")
    click.echo(f"  Persist Directory: {config.persist_directory}")
    click.echo(f"  Collection: {config.collection_name}")
    if manifest.exists:
        click.echo(f"  Indexed ZIM Files: {len(manifest.zims)}")
        click.echo(f"  Total Chunks: {manifest.total_chunks()}")
    click.echo(f"  LLM: {config.llm_provider} / {config.llm_model or DEFAULT_LLM_MODELS.get(config.llm_provider, '(not set)')}")

    if not manifest.exists:
        click.echo("\n💡 Tip: Run 'python zim_rag.py build' to create the vector database")


@cli.command()
@click.argument("zim_name")
@click.pass_context
def remove(ctx, zim_name):
    """Remove a ZIM file's chunks from the vector database."""
    config: Config = ctx.obj["config"]
    manifest = Manifest.load(config)
    if zim_name not in manifest.zims:
        raise click.ClickException(f"{zim_name} is not in the vector database")
    store = open_store(config, manifest.vector_db_type)
    store.delete_source(zim_name)
    del manifest.zims[zim_name]
    manifest.save()
    click.echo(f"Removed {zim_name} from the vector database.")


@cli.command()
@click.option(
    "--zim-file", "zim_files", multiple=True, help="ZIM file to export; repeatable. Default: all"
)
@click.option("--output", default="zim_articles.json", help="Output file")
@click.option("--limit", default=None, type=int, help="Limit number of articles per ZIM file")
@click.option(
    "--format", "fmt", type=click.Choice(["json", "jsonl"]), default=None,
    help="Output format [default: from the file extension]",
)
@click.option("--html", "raw_html", is_flag=True, help="Export raw HTML instead of clean text")
@click.pass_context
def export(ctx, zim_files, output, limit, fmt, raw_html):
    """Export articles from the ZIM library to JSON or JSON Lines."""
    config: Config = ctx.obj["config"]
    library = ZIMLibraryManager(config.zim_library_path)
    targets = library.select(zim_files)
    fmt = fmt or ("jsonl" if output.endswith(".jsonl") else "json")
    processor = TextProcessor(config.chunk_size, config.chunk_overlap, min_length=0)

    count = 0
    with open(output, "w", encoding="utf-8") as f:
        if fmt == "json":
            f.write("[\n")
        for zim_path in targets:
            exported = 0
            for _, article in ZIMReader(zim_path).iter_articles():
                if limit is not None and exported >= limit:
                    break
                if not raw_html:
                    article.content = processor.clean_html(article.content)
                record = json.dumps(article.to_dict(), ensure_ascii=False)
                if fmt == "json":
                    f.write((",\n" if count else "") + record)
                else:
                    f.write(record + "\n")
                exported += 1
                count += 1
        if fmt == "json":
            f.write("\n]\n")

    click.echo(f"Exported {count} articles from {len(targets)} ZIM file(s) to {output}")


@cli.command("list-zim")
@click.pass_context
def list_zim(ctx):
    """List all ZIM files in the library."""
    config: Config = ctx.obj["config"]
    library = ZIMLibraryManager(config.zim_library_path)
    manifest = Manifest.load(config)

    if not library.zim_files:
        click.echo(f"No ZIM files found in library: {config.zim_library_path}")
        click.echo("\n💡 To add ZIM files:")
        click.echo("1. Download ZIM files from https://library.kiwix.org/")
        click.echo("2. Place them in the zim_library directory")
        return

    click.echo(f"ZIM Library: {config.zim_library_path}")
    click.echo(f"Found {len(library.zim_files)} ZIM files:\n")
    for i, zim_path in enumerate(library.zim_files, 1):
        size_mb = zim_path.stat().st_size / (1024 * 1024)
        status = manifest.zims.get(zim_path.name, {}).get("status", "not indexed")
        click.echo(f"{i:2d}. 📁 {zim_path.name}")
        click.echo(f"      Size: {size_mb:.1f} MB · {status.replace('_', ' ')}\n")

    click.echo("💡 Use 'python zim_rag.py build --zim-file <filename>' to process specific files")
    click.echo("💡 Use 'python zim_rag.py build' to process all files")


if __name__ == "__main__":
    cli()
