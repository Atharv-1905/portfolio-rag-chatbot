
import chromadb
from pathlib import Path


class VectorStore:

    def __init__(self):
        self.db_path = Path("chroma_db")

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name="portfolio_knowledge",
            metadata={
                "description": "Atharv Bhosale Portfolio RAG",
                "hnsw:space": "cosine"
            }
        )

    # Store or update portfolio chunks
    def add_chunks(self, chunks, embeddings):

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match!"
            )

        if not chunks:
            return

        ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        documents = [
            chunk["text"]
            for chunk in chunks
        ]

        metadatas = [
            {
                "section": chunk["section"],
                "title": chunk["title"],
                "source": chunk["source"]
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print("Documents stored successfully!")

    # Semantic similarity search
    def search(self, query_embedding, n_results=3):

        total = self.collection.count()

        if total == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    # Retrieve all documents belonging to a section
    def get_by_section(self, section):

        results = self.collection.get(
            where={
                "section": section
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        records = []

        for document, metadata in zip(
            results["documents"],
            results["metadatas"]
        ):

            records.append({
                "text": document,
                "metadata": metadata,
                "distance": None
            })

        return records

    # Return total number of stored documents
    def count(self):
        return self.collection.count()


# Standalone testing
if __name__ == "__main__":

    store = VectorStore()

    print("\nChromaDB initialized successfully!")
    print("Database location:", store.db_path.resolve())
    print("Collection:", store.collection.name)
    print("Current document count:", store.count())

    # Test complete project retrieval
    projects = store.get_by_section("projects")

    print("\n========== PROJECT RETRIEVAL TEST ==========")

    print("Total project chunks:", len(projects))

    for project in projects:
        print("-", project["metadata"]["title"])

