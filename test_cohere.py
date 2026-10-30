
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

client = cohere.ClientV2(
    api_key=os.getenv("COHERE_API_KEY")
)

response = client.embed(
    model=os.getenv("COHERE_EMBED_MODEL"),
    texts=["Atharv is a Data Science and Machine Learning student."],
    input_type="search_document",
    embedding_types=["float"]
)

embedding = response.embeddings.float[0]

print("Cohere API connected successfully!")
print("Embedding dimensions:", len(embedding))
print("First 5 values:", embedding[:5])
