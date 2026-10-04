import os

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="university_knowledge"
)


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


def retrieve_chunks(question, n_results=3):
    query_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    return results


def answer_question_without_guardrail(question):
    results = retrieve_chunks(question)

    documents = results["documents"][0]

    context = "\n\n".join(
        f"- {document}" for document in documents
    )

    # Baseline prompt: no explicit knowledge-base refusal rule.
    prompt = f"""
You are a university information assistant.

Answer the user's question using the context below.

Context:
{context}

Question:
{question}
"""

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
        generation_config={
            "thinking_level": "low"
        },
    )

    return interaction.output_text.strip()


evaluation_questions = [
    "What undergraduate programs does the university offer?",
    "What resources does the university library provide?",
    "How can students get help with internships?",
    "How much is the university tuition fee?",
    "What is the university's ranking in Pakistan?",
]


print("=" * 70)
print("BASELINE EVALUATION - BEFORE GUARDRAIL")
print("=" * 70)

for index, question in enumerate(evaluation_questions, start=1):
    print(f"\nTest {index}")
    print("-" * 70)
    print("Question:", question)

    answer = answer_question_without_guardrail(question)

    print("AI answer:", answer)

print("\n" + "=" * 70)
print("BASELINE EVALUATION COMPLETE")
print("=" * 70)