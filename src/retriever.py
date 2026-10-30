
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from reranker import Reranker


class Retriever:

    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.reranker = Reranker()

    def retrieve(self, question, top_k=3):

        if not question.strip():
            raise ValueError("Question cannot be empty!")

        question_lower = " ".join(
            question.lower().replace("'", "").replace("?", "").split()
        )

        project_phrases = [
            "all projects",
            "all the projects",
            "all project",
            "all the project",
            "project titles",
            "list projects",
            "list all",
            "which projects",
            "what projects",
            "every project",
            "give me projects",
            "give me all projects",
            "give me all the projects",
            "show all projects",
            "show me all projects"
        ]

        portfolio_phrases = [
            "everything about atharv",
            "all portfolio details",
            "entire portfolio",
            "all information about atharv"
        ]

        # Complete project retrieval
        if any(
            phrase in question_lower
            for phrase in project_phrases
        ):

            print("\nRetrieval mode: Complete Projects")

            return self.vector_store.get_by_section("projects")

        # Complete portfolio retrieval
        if any(
            phrase in question_lower
            for phrase in portfolio_phrases
        ):

            print("\nRetrieval mode: Complete Portfolio")

            sections = [
                "projects",
                "experience",
                "research",
                "achievements",
                "skills",
                "about"
            ]

            all_results = []

            for section in sections:
                all_results.extend(
                    self.vector_store.get_by_section(section)
                )

            return all_results

        # Semantic retrieval
        print("\nRetrieval mode: Semantic Search")

        query_embedding = self.embedding_generator.generate_embeddings(
            [question],
            input_type="search_query"
        )[0]

        # Retrieve more candidates than the final required count
        candidate_results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=8
        )

        documents = []
        retrieved_documents = candidate_results["documents"][0]
        metadatas = candidate_results["metadatas"][0]
        distances = candidate_results["distances"][0]

        for document, metadata, distance in zip(
            retrieved_documents,
            metadatas,
            distances
        ):

            documents.append({
                "text": document,
                "metadata": metadata,
                "distance": distance
            })

        # Rerank candidates using Cohere
        print("Reranking candidate documents...")

        ranked_documents = self.reranker.rerank(
            question=question,
            documents=documents,
            top_n=top_k
        )

        return ranked_documents


if __name__ == "__main__":

    retriever = Retriever()

    test_questions = [
        "Which statistical methods are used in Atharv's model monitoring project?",
        "What is Atharv's experience with RAG?",
        "What is Atharv's university?"
    ]

    for question in test_questions:

        print("\nQuestion:", question)

        results = retriever.retrieve(
            question,
            top_k=3
        )

        for index, result in enumerate(results, 1):

            print(f"\nResult {index}")
            print("Title:", result["metadata"]["title"])
            print("Relevance score:", round(result["rerank_score"], 4))

