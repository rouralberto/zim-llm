"""Tests for zim_rag. They build small ZIM files with libzim and use a fake
embedder, so they run offline and without downloading models."""

import hashlib
import json
import re

import click
import numpy as np
import pytest
from click.testing import CliRunner
from libzim.writer import Creator, Hint, Item, StringProvider

import zim_rag

ARTICLES = {
    "Ray_Charles": (
        "Ray Charles",
        """<html><head><title>Ray Charles</title><style>.x{color:red}</style></head><body>
        <h1>Ray Charles</h1>
        <table class="infobox"><tr><th>Born</th><td>September 23, 1930<br>Albany, Georgia</td></tr></table>
        <p>Ray Charles was an American singer and pianist known as <a href="Soul">the Genius</a>
        of soul music.<sup class="reference">[1]</sup></p>
        <p>He pioneered soul music by combining blues, gospel and jazz.</p>
        <script>var tracking = "should not be indexed";</script>
        <div class="navbox">Navigation box noise</div>
        <ol class="references"><li>Reference noise</li></ol>
        <div class="zim-footer">This article is issued from Wikipedia.</div>
        </body></html>""",
    ),
    "Aspirin": (
        "Aspirin",
        """<html><body><h1>Aspirin</h1>
        <p>Aspirin is a medication used to reduce pain, fever and inflammation.
        Aspirin is also used long term to help prevent heart attacks and strokes.</p>
        <p>Common side effects of aspirin include an upset stomach.</p></body></html>""",
    ),
    "Stub": ("Stub", "<html><body><p>Too short.</p></body></html>"),
}


class HtmlItem(Item):
    def __init__(self, path, title, content, mimetype="text/html", front=True):
        super().__init__()
        self.path, self.title, self.content = path, title, content
        self.mimetype, self.front = mimetype, front

    def get_path(self):
        return self.path

    def get_title(self):
        return self.title

    def get_mimetype(self):
        return self.mimetype

    def get_contentprovider(self):
        return StringProvider(self.content)

    def get_hints(self):
        return {Hint.FRONT_ARTICLE: self.front}


def make_zim(path, title="Test ZIM", articles=ARTICLES):
    with Creator(str(path)).config_indexing(False, "eng") as creator:
        creator.set_mainpath(next(iter(articles)))
        for article_path, (article_title, html) in articles.items():
            creator.add_item(HtmlItem(article_path, article_title, html))
        creator.add_item(
            HtmlItem("_res_/style.css", "", "body { color: red; } " * 20, "text/css", False)
        )
        creator.add_item(
            HtmlItem("_res_/app.js", "", "console.log('x'); " * 20, "text/javascript", False)
        )
        creator.add_redirection("Ray", "Ray", next(iter(articles)), {Hint.FRONT_ARTICLE: False})
        for key, value in {
            "Title": title,
            "Language": "eng",
            "Creator": "Tests",
            "Date": "2026-01-01",
            "Description": "A test ZIM",
        }.items():
            creator.add_metadata(key, value)
    return path


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder."""

    model_name = "fake-model"
    dimension = 128

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
                vectors[row, bucket] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    def embed_documents(self, texts):
        return self.embed(texts)

    def embed_query(self, text):
        return self.embed([text])[0]


@pytest.fixture
def library(tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    make_zim(lib / "first.zim", "First ZIM")
    make_zim(lib / "second.zim", "Second ZIM")
    return lib


@pytest.fixture(params=["chroma", "faiss"])
def config(request, tmp_path, library, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(zim_rag, "load_embedder", lambda *a, **k: FakeEmbedder())
    cfg = zim_rag.Config(
        zim_library_path=str(library),
        persist_directory=str(tmp_path / "db"),
        vector_db_type=request.param,
        embedding_model="fake-model",
        min_article_length=50,
        chunk_size=200,
        chunk_overlap=20,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.__dict__))
    cfg.path = str(path)
    return cfg


def run_cli(config, *args):
    result = CliRunner().invoke(zim_rag.cli, ["--config", config.path, *args])
    return result


# --------------------------------------------------------------------------- #


def test_reader_metadata_is_decoded(library):
    meta = zim_rag.ZIMReader(library / "first.zim").get_metadata()
    assert meta["filename"] == "first.zim"
    assert meta["title"] == "First ZIM"
    assert meta["language"] == "eng"
    assert meta["creator"] == "Tests"


def test_reader_yields_only_html_articles(library):
    articles = [a for _, a in zim_rag.ZIMReader(library / "first.zim").iter_articles(progress=False)]
    assert sorted(a.url for a in articles) == ["Aspirin", "Ray_Charles", "Stub"]
    assert all(a.source_zim == "first.zim" for a in articles)


def test_clean_html_removes_noise_and_keeps_structure():
    text = zim_rag.TextProcessor().clean_html(ARTICLES["Ray_Charles"][1])
    assert "Ray Charles was an American singer and pianist known as the Genius of soul music." in text
    assert "Born | September 23, 1930 Albany, Georgia" in text
    for noise in ("tracking", "Navigation box", "Reference noise", "[1]", "color:red", "issued from"):
        assert noise not in text
    # Block elements stay separate paragraphs instead of being glued together.
    assert "Ray Charles\n\nBorn" in text
    assert "music.\n\nHe pioneered" in text


def test_chunks_carry_metadata_and_unique_ids(library):
    processor = zim_rag.TextProcessor(chunk_size=200, chunk_overlap=20, min_length=50)
    ids = []
    for name in ("first.zim", "second.zim"):
        for _, article in zim_rag.ZIMReader(library / name).iter_articles(progress=False):
            for chunk in processor.chunk_article(article):
                assert chunk.metadata["title"] == article.title
                assert chunk.metadata["url"] == article.url
                assert chunk.metadata["source_zim"] == name
                assert not chunk.text.startswith(".")
                ids.append(chunk.id)
    # Same article paths in two ZIM files must not collide.
    assert len(ids) > 2 and len(set(ids)) == len(ids)


def test_build_query_and_incremental_update(config):
    result = run_cli(config, "build", "--zim-file", "first.zim")
    assert result.exit_code == 0, result.output

    kb = zim_rag.KnowledgeBase(config, embedder=FakeEmbedder())
    top = kb.search("aspirin pain fever medication", k=3)[0]
    assert top.title == "Aspirin"
    assert top.source_zim == "first.zim"
    assert 0 < top.score <= 1.0001

    # A second build adds the new ZIM and skips the indexed one.
    result = run_cli(config, "build")
    assert result.exit_code == 0, result.output
    manifest = zim_rag.Manifest.load(config)
    assert set(manifest.zims) == {"first.zim", "second.zim"}
    assert all(z["status"] == "complete" and z["articles"] == 2 for z in manifest.zims.values())

    store = zim_rag.open_store(config)
    assert store.count() == manifest.total_chunks()

    kb = zim_rag.KnowledgeBase(config, embedder=FakeEmbedder())
    only_second = kb.search("aspirin", k=10, source_zim="second.zim")
    assert only_second and {r.source_zim for r in only_second} == {"second.zim"}


def test_remove_and_force_rebuild(config):
    assert run_cli(config, "build").exit_code == 0
    total = zim_rag.Manifest.load(config).total_chunks()

    result = run_cli(config, "remove", "first.zim")
    assert result.exit_code == 0, result.output
    manifest = zim_rag.Manifest.load(config)
    assert set(manifest.zims) == {"second.zim"}
    assert zim_rag.open_store(config).count() == total // 2
    kb = zim_rag.KnowledgeBase(config, embedder=FakeEmbedder())
    assert kb.search("aspirin", k=5, source_zim="first.zim") == []

    result = run_cli(config, "build", "--force")
    assert result.exit_code == 0, result.output
    assert zim_rag.open_store(config).count() == total


def test_build_refuses_different_embedding_model(config):
    assert run_cli(config, "build", "--zim-file", "first.zim").exit_code == 0
    result = run_cli(config, "build", "--embedding-model", "other-model")
    assert result.exit_code != 0
    assert "--force" in result.output


def test_interrupted_build_resumes_without_duplicates(config, library):
    manifest = zim_rag.Manifest.load(config)
    manifest.initialise(config, "fake-model")
    store = zim_rag.open_store(config)

    class InterruptingEmbedder(FakeEmbedder):
        def embed(self, texts):
            if self.calls == 1:
                raise KeyboardInterrupt
            return super().embed(texts)

    indexer = zim_rag.Indexer(config, store, InterruptingEmbedder(), manifest)
    indexer.flush_chunks = 1
    indexer.checkpoint_seconds = 0
    with pytest.raises(KeyboardInterrupt):
        indexer.index_zim(library / "first.zim")

    state = zim_rag.Manifest.load(config).zims["first.zim"]
    assert state["status"] == "in_progress"
    assert state["articles"] == 1

    # Reopen everything as a new process would.
    manifest = zim_rag.Manifest.load(config)
    store = zim_rag.open_store(config)
    indexer = zim_rag.Indexer(config, store, FakeEmbedder(), manifest)
    assert indexer.index_zim(library / "first.zim") is True

    state = manifest.zims["first.zim"]
    assert state["status"] == "complete"
    assert state["articles"] == 2
    assert store.count() == state["chunks"]


def test_limit_restricts_articles(config):
    assert run_cli(config, "build", "--limit", "1").exit_code == 0
    manifest = zim_rag.Manifest.load(config)
    assert all(z["articles"] == 1 for z in manifest.zims.values())


def test_query_without_database_fails_cleanly(config):
    result = run_cli(config, "query", "anything")
    assert result.exit_code != 0
    assert "Run 'python zim_rag.py build' first" in result.output


def test_info_and_list_zim(config):
    run_cli(config, "build", "--zim-file", "first.zim")
    result = run_cli(config, "info")
    assert result.exit_code == 0, result.output
    assert "First ZIM" in result.output and "b'" not in result.output
    assert "indexed" in result.output

    result = run_cli(config, "list-zim")
    assert result.exit_code == 0
    assert "first.zim" in result.output and "complete" in result.output


@pytest.mark.parametrize("fmt", ["json", "jsonl"])
def test_export(config, tmp_path, fmt):
    output = tmp_path / f"out.{fmt}"
    result = run_cli(config, "export", "--zim-file", "first.zim", "--output", str(output))
    assert result.exit_code == 0, result.output
    text = output.read_text()
    records = json.loads(text) if fmt == "json" else [json.loads(l) for l in text.splitlines()]
    assert len(records) == 3
    assert all("<" not in r["content"] for r in records)


def test_rag_query_uses_sources_and_strips_thinking(config):
    assert run_cli(config, "build", "--zim-file", "first.zim").exit_code == 0

    class FakeLLM:
        def stream(self, system, prompt):
            self.prompt = prompt
            yield from ["<thi", "nk>reasoning</th", "ink>\n\n", "Aspirin reduces ", "pain [1]."]

    llm = FakeLLM()
    kb = zim_rag.KnowledgeBase(config, embedder=FakeEmbedder())
    rag = zim_rag.RAGSystem(config, knowledge_base=kb, llm=llm)
    result = rag.query("What does aspirin do?", k=2)
    assert result["answer"] == "Aspirin reduces pain [1]."
    assert result["sources"][0]["title"] == "Aspirin"
    assert "[1] Aspirin (first.zim)" in llm.prompt


# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pieces, expected",
    [
        (["Hello world"], "Hello world"),
        (["<think>x</think>Answer"], "Answer"),
        (["<", "think>abc", "</", "think>", " Answer"], "Answer"),
        (["Use a < b and <thinking", " is fine"], "Use a < b and <thinking is fine"),
        (["<think>never closed"], ""),
    ],
)
def test_strip_think_tags(pieces, expected):
    assert "".join(zim_rag.strip_think_tags(pieces)) == expected


def test_build_prompt_respects_budget():
    results = [
        zim_rag.SearchResult("x" * 1000, {"title": f"T{i}", "source_zim": "z.zim"}, 0.5)
        for i in range(10)
    ]
    prompt = zim_rag.build_prompt("Q?", results, max_chars=2500)
    assert "[1] T0 (z.zim)" in prompt and "[3] T2" in prompt
    assert "[4]" not in prompt
    assert prompt.endswith("Question: Q?")


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"embeding_model": "typo"}))
    with pytest.raises(click.ClickException, match="embeding_model"):
        zim_rag.Config.load(str(path))


def test_config_validates_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"vector_db_type": "pinecone"}))
    with pytest.raises(click.ClickException, match="vector_db_type"):
        zim_rag.Config.load(str(path))


@pytest.mark.parametrize(
    "model, query_prefix, document_prefix",
    [
        ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
        ("intfloat/multilingual-e5-small", "query: ", "passage: "),
        ("all-MiniLM-L6-v2", "", ""),
    ],
)
def test_embedding_prefixes(model, query_prefix, document_prefix):
    assert zim_rag.embedding_prefixes(zim_rag.Config(), model) == (query_prefix, document_prefix)
    custom = zim_rag.Config(embedding_query_prefix="q: ")
    assert zim_rag.embedding_prefixes(custom, model)[0] == "q: "
