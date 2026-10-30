
import os
import cohere
import json

from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:

    def __init__(self):

        self.client = cohere.ClientV2(
            api_key=os.getenv("COHERE_API_KEY")
        )

        self.model = os.getenv(
            "COHERE_EMBED_MODEL",
            "embed-english-v3.0"
        )

    def generate_embeddings(
        self,
        texts,
        input_type="search_document"
    ):

        if not texts:
            return []

        response = self.client.embed(
            model=self.model,
            texts=texts,
            input_type=input_type,
            embedding_types=["float"]
        )

        return response.embeddings.float


if __name__ == "__main__":

    # Load actual portfolio chunks
    with open(
        "data/chunks.json",
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found!")

    # Test with first two actual chunks
    sample_chunks = chunks[:2]

    texts = [
        chunk["text"]
        for chunk in sample_chunks
    ]

    generator = EmbeddingGenerator()

    embeddings = generator.generate_embeddings(
        texts,
        input_type="search_document"
    )

    print("\nCohere Embedding Test Successful!")

    print("Chunks processed:", len(embeddings))

    print(
        "Embedding dimensions:",
        len(embeddings[0])
    )

    print(
        "First 5 vector values:",
        embeddings[0][:5]
    )
