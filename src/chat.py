
import os
from groq import Groq
from dotenv import load_dotenv

from retriever import Retriever

load_dotenv()


class PortfolioChat:

    def __init__(self):

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.model = os.getenv(
            "GROQ_MODEL",
            "openai/gpt-oss-120b"
        )

        self.retriever = Retriever()

        print("Portfolio RAG Chatbot initialized!")

    def generate_answer(self, question, top_k=3):

        if not question.strip():
            return "Please enter a valid question."

        # Step 1: Retrieve relevant information
        results = self.retriever.retrieve(
            question,
            top_k=top_k
        )

        if not results:
            return "I couldn't find relevant information in the portfolio."

        # Step 2: Build context
        context_parts = []

        for result in results:

            metadata = result["metadata"]

            context_parts.append(
                f"Section: {metadata['section']}\n"
                f"Title: {metadata['title']}\n"
                f"Content: {result['text']}"
            )

        context = "\n\n".join(context_parts)

        # Step 3: System instructions
        system_prompt = """
        You are Atharv Bhosale's portfolio AI assistant.

        Answer questions using ONLY the portfolio context
        provided in the user's message.

        IMPORTANT RULES:

        1. Never invent or assume information.
        2. If information is unavailable, clearly say so.
        3. For questions asking for ALL projects, list EVERY
           unique project title present in the context.
        4. Do not omit projects when complete project context
           has been provided.
        5. Avoid listing the same project twice if it has
           multiple chunks.
        6. For questions asking for all projects, provide
           a short description of each project.
        7. Give clear, concise and natural answers.
        8. Use bullet points wherever appropriate.
        9. Refer to Atharv in the third person.
        """

        # Step 4: Generate response using Groq
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": (
                        f"Portfolio Context:\n{context}\n\n"
                        f"Question: {question}\n\n"
                        "Answer strictly using the provided context."
                    )
                }
            ],
            temperature=0.2,
            max_tokens=1200
        )

        # Step 5: Return generated answer
        answer = response.choices[0].message.content

        return answer


if __name__ == "__main__":

    chatbot = PortfolioChat()

    print("\n========== PORTFOLIO RAG CHATBOT ==========")
    print("Ask questions about Atharv's portfolio.")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        try:

            answer = chatbot.generate_answer(question)

            print("\nChatbot:", answer)
            print("\n" + "=" * 55 + "\n")

        except Exception as error:

            print("\nError generating response:", error)
            print("Please try again.\n")

