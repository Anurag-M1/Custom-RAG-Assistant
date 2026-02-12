import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import qdrant_client
from dotenv import load_dotenv
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from pypdf import PdfReader
from qdrant_client import models


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".json", ".csv"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Index custom local documents into a Qdrant collection for app.py"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing files to ingest (.txt, .md, .pdf, .json, .csv)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "custom-kb"),
        help="Qdrant collection name (default: QDRANT_COLLECTION or custom-kb)",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="FastEmbed model name; must match the model used at query time",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap in characters (default: 150)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding/upsert batch size (default: 64)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before indexing",
    )
    return parser.parse_args()


def env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)

    if path.suffix.lower() == ".json":
        raw = read_text_file(path)
        try:
            parsed = json.loads(raw)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return raw

    return read_text_file(path)


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    clean = " ".join(text.split())
    if not clean:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk-overlap must be smaller than chunk-size")

    chunks = []
    step = chunk_size - chunk_overlap
    start = 0
    length = len(clean)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(clean[start:end])
        if end == length:
            break
        start += step

    return chunks


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def collect_chunks(data_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for file_path in iter_files(data_dir):
        text = load_document(file_path)
        for chunk in chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            pairs.append((str(file_path), chunk))
    return pairs


def main():
    load_dotenv()
    args = parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"Invalid data directory: {data_dir}")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_key:
        raise SystemExit("QDRANT_URL and QDRANT_API_KEY must be set in environment/.env")

    chunks = collect_chunks(data_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if not chunks:
        raise SystemExit(
            f"No supported files/chunks found in {data_dir}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    print(f"Found {len(chunks)} chunks from {data_dir}")

    embed_model = FastEmbedEmbedding(model_name=args.embedding_model)
    request_timeout = int(float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30")))
    client = qdrant_client.QdrantClient(
        url=qdrant_url,
        api_key=qdrant_key,
        prefer_grpc=env_flag("QDRANT_PREFER_GRPC", default=False),
        timeout=request_timeout,
        check_compatibility=False,
    )

    first_embedding = embed_model.get_text_embedding(chunks[0][1])
    vector_size = len(first_embedding)
    print(f"Embedding model: {args.embedding_model} (dim={vector_size})")

    if args.recreate and client.collection_exists(collection_name=args.collection):
        client.delete_collection(collection_name=args.collection)
        print(f"Deleted existing collection: {args.collection}")

    if not client.collection_exists(collection_name=args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        print(f"Created collection: {args.collection}")

    point_id = 1
    payload_pairs = chunks

    for batch in batched(payload_pairs, args.batch_size):
        texts = [item[1] for item in batch]
        embeddings = embed_model.get_text_embedding_batch(texts)

        points = []
        for idx, (source, context) in enumerate(batch):
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embeddings[idx],
                    payload={
                        "context": context,
                        "source": source,
                    },
                )
            )
            point_id += 1

        client.upsert(collection_name=args.collection, points=points)

    print(
        f"Indexed {len(payload_pairs)} chunks into collection '{args.collection}'. "
        "Set QDRANT_COLLECTION to this name before running app.py"
    )


if __name__ == "__main__":
    main()
