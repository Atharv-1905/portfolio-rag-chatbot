
import os
import cohere
from dotenv import load_dotenv

load_dotenv()


class Reranker:

    def __init__(self):

        self.client = cohere.ClientV2(
            api_key=os.getenv("COHERE_API_KEY")
        )

        self.model = "rerank-v3.5"

    def rerank(self, question, documents, top_n=3):

        if not documents:
            return []

        # Extract document text
        texts = [
            document["text"]
            for document in documents
        ]

        # Call Cohere Rerank API
        response = self.client.rerank(
            model=self.model,
            query=question,
            documents=texts,
            top_n=min(top_n, len(texts))
        )

        ranked_documents = []

        # Map reranked results back to original documents
        for result in response.results:

            original_document = documents[result.index].copy()

            original_document["rerank_score"] = result.relevance_score

            ranked_documents.append(original_document)

        return ranked_documents


if __name__ == "__main__":

    reranker = Reranker()

    test_question = (
        "Which statistical methods are used "
        "in Atharv's model monitoring project?"
    )

    test_documents = [
        {
            "text": "Atharv uses KS Test, Chi-Square, PSI, "
                    "Wasserstein Distance and Jensen-Shannon Divergence.",
            "metadata": {
                "title": "ML Model Monitoring & Drift Detection",
                "section": "projects"
            }
        },
        {
            "text": "Atharv completed a half marathon on his birthday.",
            "metadata": {
                "title": "About Atharv",
                "section": "achievements"
            }
        },
        {
            "text": "Atharv built an Agentic Compliance Auditor "
                    "using LangGraph, FastAPI and Langfuse.",
            "metadata": {
                "title": "Agentic Compliance Auditor",
                "section": "projects"
            }
        }
    ]

    results = reranker.rerank(
        question=test_question,
        documents=test_documents,
        top_n=3
    )

    print("\n========== COHERE RERANK TEST ==========")

    for index, result in enumerate(results, 1):

        print(f"\nRank: {index}")
        print("Project:", result["metadata"]["title"])
        print("Relevance score:", round(result["rerank_score"], 4))
        print("Content:", result["text"])
