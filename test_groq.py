
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL"),
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in exactly two simple sentences."
        }
    ],
    temperature=0.3,
    max_completion_tokens=200
)

print("Groq API connected successfully!")
print("\nResponse:")
print(response.choices[0].message.content)
