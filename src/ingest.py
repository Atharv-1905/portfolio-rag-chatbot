
import json
from pathlib import Path

from embeddings import EmbeddingGenerator
from vector_store import VectorStore


# ==========================================
# CONFIGURATION
# ==========================================

CHUNKS_FILE = Path("data/chunks.json")

BATCH_SIZE = 48


# ==========================================
# LOAD PORTFOLIO CHUNKS
# ==========================================

def load_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:
        raise ValueError(
            "No portfolio chunks available!"
        )

    # Validate required fields
    required_fields = [
        "chunk_id",
        "section",
        "title",
        "source",
        "text"
    ]

    for index, chunk in enumerate(chunks):

        missing = [
            field
            for field in required_fields
            if field not in chunk
        ]

        if missing:
            raise ValueError(
                f"Chunk {index} missing fields: {missing}"
            )

        if not chunk["text"].strip():
            raise ValueError(
                f"Chunk {chunk['chunk_id']} is empty"
            )

    # Ensure IDs are unique
    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate chunk IDs detected!"
        )

    return chunks


# ==========================================
# INGESTION PIPELINE
# ==========================================

def ingest():

    print("\n========== RAG INGESTION STARTED ==========")

    # 1. Load documents
    chunks = load_chunks()

    print("Total chunks loaded:", len(chunks))

    # 2. Initialize API and vector database
    embedding_generator = EmbeddingGenerator()

    vector_store = VectorStore()

    # 3. Process chunks in batches
    total_batches = (
        len(chunks) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start in enumerate(
        range(0, len(chunks), BATCH_SIZE),
        start=1
    ):

        batch = chunks[
            start:start + BATCH_SIZE
        ]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        print(
            f"\nProcessing batch "
            f"{batch_number}/{total_batches}"
        )

        # 4. Generate document embeddings
        embeddings = (
            embedding_generator.generate_embeddings(
                texts,
                input_type="search_document"
            )
        )

        # 5. Store vectors and metadata
        vector_store.add_chunks(
            batch,
            embeddings
        )

        print(
            "Documents processed:",
            len(batch)
        )

    # 6. Verify final collection count
    final_count = vector_store.count()

    print("\n========== INGESTION COMPLETED ==========")

    print("Chunks loaded:", len(chunks))
    print("Documents in ChromaDB:", final_count)

    if final_count < len(chunks):
        raise RuntimeError(
            "Some documents may not have been stored!"
        )

    print("Knowledge base is ready!")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":

    ingest()
