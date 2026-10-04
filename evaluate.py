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


def answer_question(question):
    results = retrieve_chunks(question)

    documents = results["documents"][0]
    distances = results["distances"][0]

    # Guardrail 1:
    # Refuse to answer when the retrieved context is too distant
    # from the user's question.
    best_distance = distances[0]
    print(f"Best retrieval distance: {best_distance:.4f}")

    if best_distance > 0.70:
        return "I don't have enough information in the knowledge base to answer that question."

    context = "\n\n".join(
        f"- {document}" for document in documents
    )

    # Guardrail 2:
    # The model is explicitly restricted to the retrieved context.
    prompt = f"""
You are a university information assistant.

Answer the user's question using ONLY the information provided
in the context below.

If the answer is not fully supported by the context, say exactly:
"I don't have enough information in the knowledge base to answer that question."

Do not invent, assume, estimate, or add facts that are not present
in the context.

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


# Evaluation dataset.
# The questions intentionally cover:
# 1. Supported information
# 2. Partially supported information
# 3. Completely unsupported information

evaluation_dataset = [
    {
        "question": "What undergraduate programs does the university offer?",
        "type": "supported",
        "expected": "Answer should mention computer science, business, engineering, and information technology.",
    },
    {
        "question": "What resources does the university library provide?",
        "type": "supported",
        "expected": "Answer should mention textbooks, research journals, digital databases, and quiet study areas.",
    },
    {
        "question": "How can students get help with internships?",
        "type": "supported",
        "expected": "Answer should mention the career center, career counseling, job preparation workshops, and internship guidance.",
    },
    {
        "question": "How much money can a student receive from a merit-based scholarship?",
        "type": "partially_supported",
        "expected": "The knowledge base mentions merit-based scholarships but does not provide an amount. The assistant should refuse to invent an amount.",
    },
    {
        "question": "How much does university accommodation cost per semester?",
        "type": "partially_supported",
        "expected": "The knowledge base mentions accommodation but does not provide a cost. The assistant should refuse to invent a price.",
    },
    {
        "question": "How much is the university tuition fee?",
        "type": "unsupported",
        "expected": "The assistant should state that it does not have enough information.",
    },
    {
        "question": "What is the university's ranking in Pakistan?",
        "type": "unsupported",
        "expected": "The assistant should state that it does not have enough information.",
    },
]


def is_correct(test, actual):
    refusal = "I don't have enough information"

    if test["type"] in ["partially_supported", "unsupported"]:
        return refusal in actual

    if test["type"] == "supported":
        return refusal not in actual

    return False


print("=" * 70)
print("RAG EVALUATION - GUARDRAIL VERSION")
print("=" * 70)

passed = 0

for index, test in enumerate(evaluation_dataset, start=1):
    print(f"\nTest {index}")
    print("-" * 70)

    print("Question:", test["question"])
    print("Type:", test["type"])
    print("Expected behavior:", test["expected"])

    actual = answer_question(test["question"])

    correct = is_correct(test, actual)

    print("AI answer:", actual)
    print("Result:", "PASS" if correct else "FAIL")

    if correct:
        passed += 1


total = len(evaluation_dataset)
failed = total - passed
accuracy = (passed / total) * 100

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(f"Total tests: {total}")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
print(f"Evaluation score: {accuracy:.1f}%")

if passed == total:
    print("Overall result: PASS")
else:
    print("Overall result: REVIEW FAILURES")