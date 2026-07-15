"""Load and run three common, file-based information-retrieval benchmarks.

Supported layouts are BEIR JSONL/TREC-qrels, MIRACL JSONL.GZ/TSV, and
TREC or MS MARCO TSV collections.  The loader keeps the corpus on disk and
streams it into HKM so large corpora can be bounded with ``max_documents``.

Example:
    data = load_benchmark("beir", "data/beir", dataset="scifact")
    print(data.queries, data.qrels)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .. import Searcher, build_search_index_from_documents
from .quality_benchmark import evaluate_run


BENCHMARKS = ("beir", "miracl", "trec")
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


# Hold paths and judgments while leaving the corpus itself streamable.
@dataclass(frozen=True)
class BenchmarkData:
    name: str
    dataset: str
    split: str
    corpus_path: Path
    queries: Mapping[str, str]
    qrels: Mapping[str, Mapping[str, float]]

    # Stream benchmark records in HKM's document format.
    #
    # Parameters:
    #   max_documents (int | None): Optional corpus limit; zero means no limit.
    #
    # Returns:
    #   (Iterator[dict[str, object]]): Text and stable benchmark ID metadata.
    def documents(self, max_documents: int | None = None) -> Iterator[dict[str, object]]:
        if self.name == "trec":
            records = _trec_records(self.corpus_path)
            limit = None if not max_documents else max_documents
            for index, record in enumerate(records):
                if limit is not None and index >= limit:
                    break
                document_id = str(record["_id"])
                yield {
                    "text": str(record["text"]),
                    "metadata": {"source_path": document_id, "source_id": document_id, "source_type": "trec"},
                }
            return
        limit = None if not max_documents else max_documents
        for index, record in enumerate(_corpus_records(self.corpus_path)):
            if limit is not None and index >= limit:
                break
            document_id = str(record.get("_id", record.get("docid", record.get("id", ""))))
            text = str(record.get("text", "")).strip()
            title = str(record.get("title", "")).strip()
            if not document_id or not text:
                continue
            yield {
                "text": f"{title}\n{text}" if title else text,
                "metadata": {
                    "source_path": document_id,
                    "source_id": document_id,
                    "title": title,
                    "source_type": self.name,
                },
            }


# Open plain or gzip-compressed UTF-8 text.
#
# Parameters:
#   path (Path): Input path.
#
# Returns:
#   (Iterable[str]): Lines without trailing newlines.
def _lines(path: Path) -> Iterable[str]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        yield from handle


# Decode JSONL records from a benchmark corpus.
#
# Parameters:
#   path (Path): JSONL or JSONL.GZ path.
#
# Returns:
#   (Iterator[dict[str, Any]]): Non-empty JSON objects.
def _records(path: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(_lines(path), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")
        yield record


# Stream one JSONL file or all numbered MIRACL corpus shards in order.
#
# Parameters:
#   path (Path): Corpus file or MIRACL shard directory.
#
# Returns:
#   (Iterator[dict[str, Any]]): Corpus records.
def _corpus_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_file():
        yield from _records(path)
        return
    shards = sorted(
        path.glob("docs-*.jsonl.gz"),
        key=lambda item: int(item.stem.split("-")[-1].split(".")[0]),
    )
    if not shards:
        raise FileNotFoundError(f"no corpus shards under {path}")
    for shard in shards:
        yield from _records(shard)


# Select one named file, first by exact relative path and then recursively.
#
# Parameters:
#   root (Path): Search root.
#   names (Iterable[str]): Exact filenames to prefer.
#   required_tokens (Iterable[str]): Lowercase tokens for fallback matching.
#
# Returns:
#   (Path): Selected file path.
def _find_file(root: Path, names: Iterable[str], required_tokens: Iterable[str]) -> Path:
    exact = {name.lower() for name in names}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.lower() in exact:
            return path
    tokens = tuple(token.lower() for token in required_tokens)
    candidates = [
        path for path in root.rglob("*")
        if path.is_file() and all(token in path.name.lower() for token in tokens)
    ]
    if candidates:
        return sorted(candidates)[0]
    raise FileNotFoundError(f"could not find benchmark file under {root}: {tuple(names)}")


# Parse a tab-separated query file, accepting optional headers.
#
# Parameters:
#   path (Path): Query TSV path.
#
# Returns:
#   (dict[str, str]): Query text keyed by query ID.
def _queries(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for line in _lines(path):
        fields = line.rstrip("\n").split("\t", 1)
        if len(fields) < 2 or fields[0].lower() in {"qid", "query-id", "query_id"}:
            continue
        output[fields[0]] = fields[1]
    if not output:
        raise ValueError(f"query file contains no queries: {path}")
    return output


# Parse three- or four-column TREC qrels, retaining graded labels.
#
# Parameters:
#   path (Path): Whitespace-separated qrels path.
#
# Returns:
#   (dict[str, dict[str, float]]): Query-to-document judgments.
def _qrels(path: Path) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for line in _lines(path):
        fields = line.split()
        if len(fields) >= 4:
            query_id, document_id, grade = fields[0], fields[2], fields[3]
        elif len(fields) == 3:
            query_id, document_id, grade = fields
        else:
            continue
        try:
            value = float(grade)
        except ValueError:
            continue
        if document_id in output.get(query_id, {}):
            raise ValueError(f"duplicate judgment for {query_id}/{document_id} in {path}")
        output.setdefault(query_id, {})[document_id] = value
    if not output:
        raise ValueError(f"qrels file contains no judgments: {path}")
    return output


# Locate a BEIR dataset directory and load its standard files.
#
# Parameters:
#   root (str | Path): Dataset root or directory containing the dataset.
#   dataset (str): BEIR dataset name, such as scifact or nfcorpus.
#   split (str): Qrels split, normally test.
#
# Returns:
#   (BenchmarkData): Streamable corpus and loaded judgments.
def load_beir(root: str | Path, dataset: str = "scifact", split: str = "test") -> BenchmarkData:
    base = Path(root).expanduser()
    if (base / dataset).is_dir():
        base = base / dataset
    corpus = _find_file(base, ("corpus.jsonl",), ("corpus",))
    queries = _find_file(base, ("queries.jsonl",), ("queries",))
    qrels = _find_file(base, (f"{split}.tsv", f"qrels.{split}.tsv"), ("qrels", split))
    query_map = {
        str(record.get("_id", record.get("id", ""))): str(record.get("text", ""))
        for record in _records(queries)
        if record.get("_id", record.get("id")) is not None
    }
    return BenchmarkData("beir", dataset, split, corpus, query_map, _qrels(qrels))


# Locate a MIRACL language/split directory and load its standard files.
#
# Parameters:
#   root (str | Path): MIRACL dataset root.
#   language (str): ISO language code, such as en or es.
#   split (str): train or dev.
#
# Returns:
#   (BenchmarkData): Streamable corpus and loaded judgments.
def load_miracl(root: str | Path, language: str = "en", split: str = "dev") -> BenchmarkData:
    base = Path(root).expanduser()
    prefix = f"miracl-v1.0-{language}"
    corpus_dirs = sorted(
        path for path in base.rglob(f"miracl-corpus-v1.0-{language}")
        if path.is_dir() and any(path.glob("docs-*.jsonl.gz"))
    )
    corpus = corpus_dirs[0] if corpus_dirs else _find_file(
        base,
        (f"{prefix}.jsonl.gz", "corpus.jsonl.gz", "corpus.jsonl"),
        ("corpus", language),
    )
    topics = _find_file(base, (), ("topic", language, split))
    qrels = _find_file(base, (), ("qrel", language, split))
    return BenchmarkData("miracl", language, split, corpus, _queries(topics), _qrels(qrels))


# Locate a TREC or MS MARCO TSV collection and load its qrels.
#
# Parameters:
#   root (str | Path): Directory containing collection, queries, and qrels.
#   split (str): Optional split token used when selecting qrels.
#   corpus_path (str | Path | None): Explicit collection path.
#   queries_path (str | Path | None): Explicit query path.
#   qrels_path (str | Path | None): Explicit qrels path.
#
# Returns:
#   (BenchmarkData): Streamable TSV corpus and loaded judgments.
def load_trec(
    root: str | Path,
    split: str = "test",
    corpus_path: str | Path | None = None,
    queries_path: str | Path | None = None,
    qrels_path: str | Path | None = None,
) -> BenchmarkData:
    base = Path(root).expanduser()
    corpus = Path(corpus_path) if corpus_path else _find_file(base, ("collection.tsv", "collection.txt"), ("collection",))
    queries = Path(queries_path) if queries_path else _find_file(base, ("queries.tsv", "queries.txt"), ("quer",))
    qrels = Path(qrels_path) if qrels_path else _find_file(
        base,
        (f"qrels.{split}", f"qrels.{split}.txt", f"qrels.{split}.tsv", "qrels.txt", "qrels.tsv"),
        ("qrel",),
    )
    return BenchmarkData("trec", "trec", split, corpus, _queries(queries), _qrels(qrels))


# Stream records from a two-column TREC/MS MARCO collection TSV.
#
# Parameters:
#   path (Path): Collection path.
#
# Returns:
#   (Iterator[dict[str, object]]): HKM document records.
def _trec_records(path: Path) -> Iterator[dict[str, object]]:
    for line in _lines(path):
        fields = line.rstrip("\n").split("\t", 1)
        if len(fields) < 2 or fields[0].lower() in {"pid", "docid", "document-id"}:
            continue
        yield {"_id": fields[0], "text": fields[1]}


# Return a document stream for all supported benchmark corpus formats.
def _benchmark_documents(data: BenchmarkData, max_documents: int | None) -> Iterator[dict[str, object]]:
    yield from data.documents(max_documents)


# Download and safely extract a standard benchmark archive.
#
# Parameters:
#   url (str): Archive URL.
#   destination (str | Path): Directory receiving the archive and files.
#
# Returns:
#   (Path): Destination directory.
def download_archive(url: str, destination: str | Path) -> Path:
    output = Path(destination).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / Path(url.split("?", 1)[0]).name
    partial = archive.with_name(archive.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(partial, archive)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            root = output.resolve()
            for member in handle.infolist():
                target = (output / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"archive path escapes destination: {member.filename}")
            handle.extractall(output)
    elif archive.name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as handle:
            root = output.resolve()
            for member in handle.getmembers():
                target = (output / member.name).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f"archive path escapes destination: {member.name}")
            handle.extractall(output)
    return output


# Download the official BEIR archive for one dataset.
#
# Parameters:
#   root (str | Path): Destination root.
#   dataset (str): BEIR dataset name.
#
# Returns:
#   (Path): Destination root.
def download_beir(root: str | Path, dataset: str = "scifact") -> Path:
    return download_archive(BEIR_URL.format(dataset=dataset), root)


# Load one supported benchmark by its stable short name.
#
# Parameters:
#   name (str): beir, miracl, or trec.
#   root (str | Path): Local benchmark data root.
#   dataset (str): BEIR dataset name.
#   language (str): MIRACL language code.
#   split (str): Judgment split.
#
# Returns:
#   (BenchmarkData): Loaded benchmark.
def load_benchmark(
    name: str,
    root: str | Path,
    dataset: str = "scifact",
    language: str = "en",
    split: str = "test",
) -> BenchmarkData:
    if name == "beir":
        return load_beir(root, dataset, split)
    if name == "miracl":
        return load_miracl(root, language, split)
    if name == "trec":
        return load_trec(root, split)
    raise ValueError(f"benchmark must be one of {BENCHMARKS}")


# Build HKM, run every selected query, and calculate standard retrieval metrics.
#
# Parameters:
#   data (BenchmarkData): Loaded benchmark.
#   index_root (str | Path): Temporary or persistent HKM index root.
#   mode (str): HKM search mode.
#   k (int): Evaluation cutoff.
#   max_queries (int | None): Optional query limit; zero means no limit.
#   max_documents (int | None): Optional corpus limit; zero means no limit.
#   workers (int): HKM worker count.
#
# Returns:
#   (dict[str, Any]): Dataset, build, run, and metric report.
def run_benchmark(
    data: BenchmarkData,
    index_root: str | Path,
    mode: str = "token",
    k: int = 10,
    max_queries: int | None = None,
    max_documents: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    if k < 1 or workers < 1:
        raise ValueError("k and workers must be positive")
    query_ids = [query_id for query_id in data.queries if query_id in data.qrels]
    if max_queries:
        query_ids = query_ids[:max_queries]
    qrels = {query_id: data.qrels[query_id] for query_id in query_ids}
    started = time.perf_counter()
    build_search_index_from_documents(
        str(index_root),
        _benchmark_documents(data, max_documents),
        num_workers=workers,
        max_k=8,
    )
    build_seconds = time.perf_counter() - started
    searcher = Searcher.from_index_root(str(index_root))
    ingest_report = Path(index_root) / "docs" / "worker_0000" / "ingest_report.json"
    indexed_documents = 0
    if ingest_report.exists():
        indexed_documents = int(json.loads(ingest_report.read_text(encoding="utf-8")).get("indexed", 0))
    run: dict[str, list[str]] = {}
    search_seconds = 0.0
    for query_id in qrels:
        query_started = time.perf_counter()
        result = searcher.search({"mode": mode, "text": data.queries[query_id], "top_k": k})
        search_seconds += time.perf_counter() - query_started
        run[query_id] = [hit.source_path for hit in result.docs]
    return {
        "benchmark": data.name,
        "dataset": data.dataset,
        "split": data.split,
        "corpus_path": str(data.corpus_path),
        "queries": len(qrels),
        "judged_queries": len(data.qrels),
        "judgments": sum(len(items) for items in qrels.values()),
        "indexed_documents": indexed_documents,
        "max_documents": max_documents or 0,
        "mode": mode,
        "k": k,
        "build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "evaluation": evaluate_run(run, qrels, k),
    }


# Command-line entry point for local benchmark execution.
def main() -> None:
    parser = argparse.ArgumentParser(description="Run BEIR, MIRACL, or TREC against HKM")
    parser.add_argument("benchmark", choices=BENCHMARKS)
    parser.add_argument("data_root")
    parser.add_argument("index_root")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--language", default="en")
    parser.add_argument("--split", default=None)
    parser.add_argument("--mode", choices=("token", "semantic", "hybrid"), default="hybrid")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--max-documents", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--download-beir", action="store_true")
    args = parser.parse_args()
    split = args.split or ("test" if args.benchmark != "miracl" else "dev")
    if args.download_beir:
        if args.benchmark != "beir":
            parser.error("--download-beir is only valid for beir")
        download_beir(args.data_root, args.dataset)
    data = load_benchmark(args.benchmark, args.data_root, args.dataset, args.language, split)
    report = run_benchmark(
        data,
        args.index_root,
        args.mode,
        args.k,
        args.max_queries,
        args.max_documents,
        args.workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


__all__ = [
    "BEIR_URL", "BENCHMARKS", "BenchmarkData", "download_archive", "download_beir",
    "load_beir", "load_benchmark", "load_miracl", "load_trec", "run_benchmark",
]


if __name__ == "__main__":
    main()
