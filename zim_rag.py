#!/usr/bin/env python3
"""
ZIM to Vector Database RAG System

This script processes ZIM files to create a vector database for LLM RAG applications.
It extracts articles from ZIM files, creates embeddings, and stores them in a vector database
for efficient semantic search and retrieval-augmented generation.

Usage:
    python zim_rag.py build --zim-file fas-military-medicine_en_2025-06.zim
    python zim_rag.py query "What are the latest treatments for PTSD?"
    python zim_rag.py info
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import click
import faiss
import numpy as np
from bs4 import BeautifulSoup
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Anthropic, Ollama, OpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Try to import zim libraries
try:
    import libzim

    zim = libzim
except ImportError:
    zim = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("zim_rag.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class DockerModelRunnerLLM(LLM):
    """Custom LLM class that uses Docker Model Runner CLI."""

    model_name: str = "ai/smollm3:Q4_K_M"

    @property
    def _llm_type(self) -> str:
        return "docker_model_runner"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Run the Docker Model Runner CLI with the given prompt."""
        try:
            # For Docker Model Runner CLI, we need to pass the prompt as a single argument
            # Escape single quotes by replacing them with escaped quotes
            escaped_prompt = prompt.replace("'", "'\"'\"'")
            cmd = ["docker", "model", "run", self.model_name, escaped_prompt]

            logger.debug(f"Running command: {' '.join(cmd)}")

            # Run the command
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                response = result.stdout.strip()
                # Clean up the response - remove any extra whitespace
                return response
            else:
                error_msg = result.stderr.strip()
                logger.error(f"Docker Model Runner CLI error: {error_msg}")
                return f"Error: {error_msg}"

        except subprocess.TimeoutExpired:
            logger.error("Docker Model Runner CLI timed out")
            return "Error: Request timed out"
        except FileNotFoundError:
            logger.error("Docker command not found. Is Docker installed and in PATH?")
            return "Error: Docker not found"
        except Exception as e:
            logger.error(f"Unexpected error running Docker Model Runner: {e}")
            return f"Error: {str(e)}"

    @property
    def _identifying_params(self) -> Dict[str, str]:
        """Return identifying parameters."""
        return {"model_name": self.model_name}


@dataclass
class Article:
    """Represents an article extracted from ZIM file."""

    url: str
    title: str
    content: str
    namespace: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def id(self) -> str:
        """Generate unique ID for the article."""
        return hashlib.md5(f"{self.url}{self.title}".encode()).hexdigest()

    def to_dict(self) -> Dict:
        """Convert article to dictionary."""
        return asdict(self)


@dataclass
class Config:
    """Configuration for the ZIM RAG system."""

    zim_library_path: str = "./zim_library"
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_type: str = "chroma"  # 'chroma' or 'faiss'
    chunk_size: int = 1000
    chunk_overlap: int = 200
    persist_directory: str = "./vector_db"
    collection_name: str = "zim_articles"
    llm_provider: str = (
        "docker_model_runner"  # 'openai', 'anthropic', 'ollama', 'docker_model_runner'
    )
    llm_model: str = "ai/smollm3:Q4_K_M"
    max_articles_per_zim: Optional[int] = None  # Limit articles per ZIM file


class ZIMReader:
    """Reader for ZIM files supporting multiple libraries."""

    def __init__(self, zim_path: str):
        self.zim_path = Path(zim_path)
        self._zim_file = None
        self._reader_type = None

        if not self.zim_path.exists():
            raise FileNotFoundError(f"ZIM file not found: {zim_path}")

        self._initialize_reader()

    def _initialize_reader(self):
        """Initialize the appropriate ZIM reader."""
        # Try libzim first (recommended)
        if zim is not None:
            try:
                self._zim_file = zim.reader.Archive(str(self.zim_path))
                self._reader_type = "libzim"
                logger.info("Using libzim for ZIM file reading")
                return
            except Exception as e:
                logger.warning(f"libzim failed: {e}")

        # Fallback to command line tools
        self._reader_type = "cli"
        logger.warning("Using CLI tools for ZIM file reading (limited functionality)")

    def get_metadata(self) -> Dict:
        """Get ZIM file metadata."""
        if self._reader_type == "libzim":
            return {
                "title": str(self._zim_file.get_metadata("Title") or ""),
                "description": str(self._zim_file.get_metadata("Description") or ""),
                "language": str(self._zim_file.get_metadata("Language") or ""),
                "creator": str(self._zim_file.get_metadata("Creator") or ""),
                "article_count": self._zim_file.article_count,
            }
        else:
            # CLI fallback - limited metadata
            return {"title": self.zim_path.stem, "article_count": 0}

    def extract_articles(self, limit: Optional[int] = None) -> List[Article]:
        """Extract articles from the ZIM file."""
        articles = []

        if self._reader_type == "libzim":
            articles = self._extract_with_libzim(limit)
        else:
            articles = self._extract_with_cli(limit)

        logger.info(f"Extracted {len(articles)} articles from ZIM file")
        return articles

    def _extract_with_libzim(self, limit: Optional[int]) -> List[Article]:
        """Extract articles using libzim."""
        articles = []
        count = 0

        for i in tqdm(
            range(self._zim_file.all_entry_count), desc="Extracting articles"
        ):
            if limit and count >= limit:
                break

            try:
                entry = self._zim_file._get_entry_by_id(i)
                if entry.is_redirect:
                    continue

                item = entry.get_item()
                # Skip non-article content (CSS, JS, images, etc.)
                if not item.mimetype.startswith("text/"):
                    continue

                content = bytes(item.content)
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="ignore")

                # Skip if content is too short or looks like metadata/scripts
                if len(content.strip()) < 100 or item.title.startswith("_"):
                    continue

                zim_article = Article(
                    url=item.path,
                    title=item.title,
                    content=content,
                    namespace="A",  # Default namespace for articles
                    metadata={"mimetype": item.mimetype},
                )
                articles.append(zim_article)
                count += 1

            except Exception as e:
                logger.warning(f"Error extracting article {i}: {e}")
                continue

        return articles

    def _extract_with_cli(self, limit: Optional[int]) -> List[Article]:
        """Extract articles using CLI tools as fallback."""
        # This is a basic implementation - would need zim-tools installed
        logger.warning(
            "CLI extraction not fully implemented. Please install zimply or libzim."
        )
        return []


class ZIMLibraryManager:
    """Manages a collection of ZIM files from a library directory."""

    def __init__(self, library_path: str):
        self.library_path = Path(library_path)
        self.zim_files = []

        if not self.library_path.exists():
            logger.warning(f"ZIM library path does not exist: {library_path}")
            self.library_path.mkdir(parents=True, exist_ok=True)

        self._scan_library()

    def _scan_library(self):
        """Scan the library directory for ZIM files."""
        zim_files = list(self.library_path.glob("*.zim"))
        self.zim_files = sorted(zim_files)  # Sort for consistent processing

        logger.info(f"Found {len(self.zim_files)} ZIM files in library:")
        for zim_file in self.zim_files:
            logger.info(f"  - {zim_file.name}")

    def get_library_info(self) -> Dict:
        """Get information about all ZIM files in the library."""
        info = {
            "library_path": str(self.library_path),
            "total_zim_files": len(self.zim_files),
            "zim_files": [],
        }

        for zim_path in self.zim_files:
            try:
                reader = ZIMReader(str(zim_path))
                metadata = reader.get_metadata()
                info["zim_files"].append(metadata)
            except Exception as e:
                logger.warning(f"Could not read metadata for {zim_path.name}: {e}")
                info["zim_files"].append(
                    {"filename": zim_path.name, "title": zim_path.stem, "error": str(e)}
                )

        return info

    def extract_all_articles(
        self, max_articles_per_zim: Optional[int] = None
    ) -> List[Article]:
        """Extract articles from all ZIM files in the library."""
        all_articles = []

        for zim_path in self.zim_files:
            logger.info(f"Processing ZIM file: {zim_path.name}")

            try:
                reader = ZIMReader(str(zim_path))
                articles = reader.extract_articles(limit=max_articles_per_zim)

                # Add source information to each article
                for article in articles:
                    article.metadata["source_zim"] = zim_path.name
                    article.metadata["source_path"] = str(zim_path)

                all_articles.extend(articles)
                logger.info(f"Extracted {len(articles)} articles from {zim_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {zim_path.name}: {e}")
                continue

        logger.info(f"Total articles extracted from all ZIM files: {len(all_articles)}")
        return all_articles

    def extract_articles_from_zim(
        self, zim_filename: str, limit: Optional[int] = None
    ) -> List[Article]:
        """Extract articles from a specific ZIM file."""
        zim_path = self.library_path / zim_filename
        if not zim_path.exists():
            raise FileNotFoundError(f"ZIM file not found: {zim_filename}")

        reader = ZIMReader(str(zim_path))
        articles = reader.extract_articles(limit=limit)

        # Add source information
        for article in articles:
            article.metadata["source_zim"] = zim_filename
            article.metadata["source_path"] = str(zim_path)

        return articles


class TextProcessor:
    """Process and chunk text for embedding."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract readable text."""
        if not html_content:
            return ""

        try:
            # Parse HTML and extract text
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)

            return text
        except Exception as e:
            logger.warning(f"Error cleaning HTML content: {e}")
            # Return original content if HTML parsing fails
            return html_content

    def process_articles(self, articles: List[Article]) -> List[Dict]:
        """Process articles into chunks suitable for embedding."""
        chunks = []

        for article in tqdm(articles, desc="Processing articles"):
            # Clean HTML content before processing
            cleaned_content = self.clean_html(article.content)

            # Skip if content is too short after cleaning
            if len(cleaned_content.strip()) < 100:
                continue

            # Split cleaned content into chunks
            content_chunks = self.text_splitter.split_text(cleaned_content)

            for i, chunk in enumerate(content_chunks):
                chunk_data = {
                    "id": f"{article.id}_chunk_{i}",
                    "content": chunk,
                    "title": article.title,
                    "url": article.url,
                    "chunk_index": i,
                    "metadata": {
                        **article.metadata,
                        "source": "zim",
                        "article_id": article.id,
                        "total_chunks": len(content_chunks),
                    },
                }
                chunks.append(chunk_data)

        logger.info(f"Created {len(chunks)} text chunks from {len(articles)} articles")
        return chunks


class VectorDatabase:
    """Vector database for storing and retrieving embeddings."""

    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = None
        self.vector_store = None
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)

    def build_from_chunks(self, chunks: List[Dict]):
        """Build vector database from text chunks."""
        if not chunks:
            raise ValueError("No chunks provided for building vector database")

        # Extract texts and metadata
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [chunk["id"] for chunk in chunks]

        logger.info(f"Creating embeddings for {len(texts)} chunks")

        embeddings = []
        batch_size = 32

        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts, show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)

        if self.config.vector_db_type == "chroma":
            self._create_chroma_store(texts, embeddings, metadatas, ids)
        elif self.config.vector_db_type == "faiss":
            self._create_faiss_store(texts, embeddings, metadatas)
        else:
            raise ValueError(
                f"Unsupported vector database type: {self.config.vector_db_type}"
            )

        logger.info(f"Vector database created with {len(texts)} documents")

    def _create_chroma_store(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict],
        ids: List[str],
    ):
        """Create ChromaDB vector store."""
        # Initialize Chroma client
        client = chromadb.PersistentClient(path=self.config.persist_directory)

        collection = client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"description": "ZIM articles for RAG"},
        )

        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Adding to ChromaDB"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size].tolist()
            batch_metadatas = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

        self.vector_store = Chroma(
            client=client,
            collection_name=self.config.collection_name,
            embedding_function=HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            ),
        )

    def _create_faiss_store(
        self, texts: List[str], embeddings: np.ndarray, metadatas: List[Dict]
    ):
        """Create FAISS vector store."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        index.add(embeddings.astype(np.float32))
        self.vector_store = FAISS(
            embedding_function=HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            ),
            index=index,
            docstore=None,
            index_to_docstore_id={},
        )

        # Save the index
        faiss.write_index(
            index, os.path.join(self.config.persist_directory, "faiss_index.idx")
        )

    def load_existing(self) -> bool:
        """Load existing vector database if available."""
        try:
            if self.config.vector_db_type == "chroma":
                persist_dir = Path(self.config.persist_directory)
                if persist_dir.exists():
                    client = chromadb.PersistentClient(path=str(persist_dir))
                    collection = client.get_collection(self.config.collection_name)
                    if collection.count() > 0:
                        self.vector_store = Chroma(
                            client=client,
                            collection_name=self.config.collection_name,
                            embedding_function=HuggingFaceEmbeddings(
                                model_name=self.config.embedding_model
                            ),
                        )
                        logger.info("Loaded existing ChromaDB vector store")
                        return True
            elif self.config.vector_db_type == "faiss":
                index_path = Path(self.config.persist_directory) / "faiss_index.idx"
                if index_path.exists():
                    index = faiss.read_index(str(index_path))
                    self.vector_store = FAISS(
                        embedding_function=HuggingFaceEmbeddings(
                            model_name=self.config.embedding_model
                        ),
                        index=index,
                        docstore=None,
                        index_to_docstore_id={},
                    )
                    logger.info("Loaded existing FAISS vector store")
                    return True
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {e}")

        return False

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search the vector database for similar documents."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        if self.config.vector_db_type == "chroma":
            # ChromaDB search
            results = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )
            return [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in results
            ]
        else:
            # FAISS search
            docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )
            return [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in docs_and_scores
            ]


class RAGSystem:
    """Retrieval-Augmented Generation system."""

    def __init__(self, config: Config):
        self.config = config
        self.vector_db = VectorDatabase(config)
        self.llm = None
        self.qa_chain = None

    def initialize_llm(self):
        """Initialize the LLM based on configuration."""
        if self.config.llm_provider == "openai":
            self.llm = OpenAI(model_name=self.config.llm_model)
        elif self.config.llm_provider == "anthropic":
            self.llm = Anthropic(model=self.config.llm_model)
        elif self.config.llm_provider == "ollama":
            self.llm = Ollama(model=self.config.llm_model)
        elif self.config.llm_provider == "docker_model_runner":
            # Use Docker Model Runner CLI directly
            self.llm = DockerModelRunnerLLM(model_name=self.config.llm_model)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")

    def setup_qa_chain(self):
        """Set up the QA chain for RAG."""
        if not self.vector_db.vector_store:
            raise ValueError("Vector database not loaded")

        if not self.llm:
            self.initialize_llm()

        retriever = self.vector_db.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        prompt_template = (
            "Use the following pieces of context from the ZIM knowledge base to answer the question at the end. "
            "If you don't know the answer based on the provided context, say that you don't know and don't try to make up an answer.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

    def query(self, question: str) -> Dict:
        """Query the RAG system."""
        if not self.qa_chain:
            self.setup_qa_chain()

        result = self.qa_chain({"query": question})

        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                    "title": doc.metadata.get("title", "Unknown"),
                    "url": doc.metadata.get("url", "Unknown"),
                }
                for doc in result["source_documents"]
            ],
        }


@click.group()
@click.option("--config", default=None, help="Path to config file")
@click.pass_context
def cli(ctx, config):
    """ZIM to Vector Database RAG System"""
    ctx.ensure_object(dict)
    if config:
        with open(config, "r") as f:
            config_data = json.load(f)
        ctx.obj["config"] = Config(**config_data)
    else:
        ctx.obj["config"] = Config()


@cli.command()
@click.option(
    "--zim-file",
    default=None,
    help="Specific ZIM file to process (optional, processes all if not specified)",
)
@click.option(
    "--limit", default=None, type=int, help="Limit number of articles per ZIM file"
)
@click.option("--force", is_flag=True, help="Force rebuild even if vector DB exists")
@click.pass_context
def build(ctx, zim_file, limit, force):
    """Build vector database from ZIM library."""
    config = ctx.obj["config"]

    logger.info(f"Building vector database from ZIM library: {config.zim_library_path}")

    vector_db = VectorDatabase(config)
    if not force and vector_db.load_existing():
        logger.info("Vector database already exists. Use --force to rebuild.")
        return

    library_manager = ZIMLibraryManager(config.zim_library_path)

    if not library_manager.zim_files:
        logger.error(f"No ZIM files found in library: {config.zim_library_path}")
        return

    library_info = library_manager.get_library_info()
    logger.info(f"Library contains {library_info['total_zim_files']} ZIM files")

    if zim_file:
        logger.info(f"Processing specific ZIM file: {zim_file}")
        try:
            articles = library_manager.extract_articles_from_zim(
                zim_file, limit=limit or config.max_articles_per_zim
            )
        except FileNotFoundError:
            logger.error(f"ZIM file not found: {zim_file}")
            return
    else:
        articles = library_manager.extract_all_articles(
            max_articles_per_zim=limit or config.max_articles_per_zim
        )

    if not articles:
        logger.error("No articles extracted from ZIM files")
        return

    processor = TextProcessor(config.chunk_size, config.chunk_overlap)
    chunks = processor.process_articles(articles)

    vector_db.build_from_chunks(chunks)

    logger.info(f"Vector database built successfully from {len(articles)} articles!")


@cli.command()
@click.argument("question")
@click.option("--k", default=5, type=int, help="Number of documents to retrieve")
@click.pass_context
def query(ctx, question, k):
    """Query the vector database."""
    config = ctx.obj["config"]

    vector_db = VectorDatabase(config)
    if not vector_db.load_existing():
        logger.error("Vector database not found. Run 'build' command first.")
        return

    results = vector_db.search(question, k=k)

    print(f"\nQuery: {question}")
    print(f"Found {len(results)} relevant documents:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"   URL: {result['metadata'].get('url', 'Unknown')}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Content: {result['content'][:300]}...")
        print()


@cli.command()
@click.argument("question")
@click.pass_context
def rag_query(ctx, question):
    """Query using RAG (Retrieval-Augmented Generation)."""
    config = ctx.obj["config"]

    rag = RAGSystem(config)

    if not rag.vector_db.load_existing():
        logger.error("Vector database not found. Run 'build' command first.")
        return

    result = rag.query(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}\n")

    print("Source documents:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"{i}. {doc['title']} ({doc['url']})")
        print(f"   {doc['content']}\n")


@cli.command()
@click.pass_context
def info(ctx):
    """Show information about the ZIM library and vector database."""
    config = ctx.obj["config"]

    library_manager = ZIMLibraryManager(config.zim_library_path)
    library_info = library_manager.get_library_info()

    print("ZIM Library Information:")
    print(f"  Library Path: {library_info['library_path']}")
    print(f"  Total ZIM Files: {library_info['total_zim_files']}")

    if library_info["zim_files"]:
        print("\nZIM Files:")
        for zim_info in library_info["zim_files"]:
            print(f"  📁 {zim_info.get('filename', 'Unknown')}")
            print(f"    Title: {zim_info.get('title', 'Unknown')}")
            print(f"    Language: {zim_info.get('language', 'Unknown')}")
            print(f"    Creator: {zim_info.get('creator', 'Unknown')}")
            print(f"    Articles: {zim_info.get('article_count', 'Unknown')}")
            if "error" in zim_info:
                print(f"    ⚠️  Error: {zim_info['error']}")
            print()

    # Load vector database
    vector_db = VectorDatabase(config)
    db_loaded = vector_db.load_existing()

    print("Vector Database Information:")
    print(f"  Status: {'✅ Loaded' if db_loaded else '❌ Not found'}")
    print(f"  Type: {config.vector_db_type}")
    print(f"  Embedding Model: {config.embedding_model}")
    print(f"  Persist Directory: {config.persist_directory}")
    print(f"  Collection: {config.collection_name}")

    if not db_loaded:
        print("\n💡 Tip: Run 'python3 zim_rag.py build' to create the vector database")


@cli.command()
@click.option(
    "--zim-file",
    default=None,
    help="Specific ZIM file to export (optional, exports all if not specified)",
)
@click.option(
    "--output", default="zim_articles.json", help="Output file for exported articles"
)
@click.option(
    "--limit", default=None, type=int, help="Limit number of articles per ZIM file"
)
@click.pass_context
def export(ctx, zim_file, output, limit):
    """Export articles from ZIM library to JSON."""
    config = ctx.obj["config"]

    library_manager = ZIMLibraryManager(config.zim_library_path)

    if not library_manager.zim_files:
        logger.error(f"No ZIM files found in library: {config.zim_library_path}")
        return

    if zim_file:
        logger.info(f"Exporting articles from: {zim_file}")
        try:
            articles = library_manager.extract_articles_from_zim(zim_file, limit=limit)
        except FileNotFoundError:
            logger.error(f"ZIM file not found: {zim_file}")
            return
    else:
        articles = library_manager.extract_all_articles(max_articles_per_zim=limit)

    if not articles:
        logger.error("No articles extracted from ZIM files")
        return

    articles_data = [article.to_dict() for article in articles]

    with open(output, "w", encoding="utf-8") as f:
        json.dump(articles_data, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Exported {len(articles)} articles from {len(library_manager.zim_files)} ZIM files to {output}"
    )


@cli.command()
@click.pass_context
def list_zim(ctx):
    """List all ZIM files in the library."""
    config = ctx.obj["config"]

    library_manager = ZIMLibraryManager(config.zim_library_path)

    if not library_manager.zim_files:
        print(f"No ZIM files found in library: {config.zim_library_path}")
        print("\n💡 To add ZIM files:")
        print("1. Download ZIM files from https://library.kiwix.org/")
        print("2. Place them in the zim_library directory")
        return

    print(f"ZIM Library: {config.zim_library_path}")
    print(f"Found {len(library_manager.zim_files)} ZIM files:\n")

    for i, zim_path in enumerate(library_manager.zim_files, 1):
        file_size = zim_path.stat().st_size / (1024 * 1024)  # MB
        print(f"{i:2d}. 📁 {zim_path.name}")
        print(f"      Size: {file_size:.1f} MB")
        print()

    print(
        "💡 Use 'python3 zim_rag.py build --zim-file <filename>' to process specific files"
    )
    print("💡 Use 'python3 zim_rag.py build' to process all files")


if __name__ == "__main__":
    cli()
