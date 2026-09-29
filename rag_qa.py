import os

import chromadb
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")

client = genai.Client(api_key=api_key)

# Create a persistent local Chroma database.
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load the RAG knowledge collection.
collection = chroma_client.get_or_create_collection(
    name="university_knowledge"
)

# Small fictional university knowledge base.
# Each item is a meaningful document chunk.
knowledge_chunks = [
    {
        "id": "chunk-1",
        "text": "The fictional university accepts undergraduate applications through its online admissions portal. Applicants must complete the application form and submit the required academic documents before the published deadline.",
        "category": "admissions",
    },
    {
        "id": "chunk-2",
        "text": "The university offers undergraduate programs in computer science, business, engineering, and information technology. Program requirements can vary by department.",
        "category": "programs",
    },
    {
        "id": "chunk-3",
        "text": "Students can apply for merit-based scholarships through the financial aid office. Scholarship decisions are based on the eligibility requirements published by the university.",
        "category": "financial_aid",
    },
    {
        "id": "chunk-4",
        "text": "The university library provides textbooks, research journals, digital databases, and quiet study areas. Students can access online academic resources using their university accounts.",
        "category": "library",
    },
    {
        "id": "chunk-5",
        "text": "The university provides student accommodation near the main campus for eligible students. Housing availability depends on capacity and the university accommodation process.",
        "category": "accommodation",
    },
    {
        "id": "chunk-6",
        "text": "Final examinations are normally held at the end of each academic semester. Students should check the official examination schedule for their courses and examination dates.",
        "category": "examinations",
    },
    {
        "id": "chunk-7",
        "text": "Students can join university clubs and societies covering academic, cultural, technical, sports, and social activities. Club membership information is available through student services.",
        "category": "student_life",
    },
    {
        "id": "chunk-8",
        "text": "The university career center helps students prepare for internships and employment by providing career counseling, job preparation workshops, and internship guidance.",
        "category": "careers",
    },
    {
        "id": "chunk-9",
        "text": "Students can contact the IT support team when they have problems with university accounts, password access, campus systems, or other technical services.",
        "category": "technical_support",
    },
    {
        "id": "chunk-10",
        "text": "The university academic office assists students with course registration, academic procedures, and general questions about their academic progress.",
        "category": "academics",
    },
]


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


if collection.count() == 0:
    print("Generating embeddings for the knowledge base...")

    for chunk in knowledge_chunks:
        embedding = get_embedding(chunk["text"])

        collection.upsert(
            ids=[chunk["id"]],
            documents=[chunk["text"]],
            embeddings=[embedding],
            metadatas=[{"category": chunk["category"]}],
        )

    print(f"Stored {len(knowledge_chunks)} knowledge chunks in Chroma.")
else:
    print(f"Using existing knowledge base with {collection.count()} chunks.")

print("Knowledge base is ready.")

def retrieve_chunks(question, n_results=3):
    query_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    return results

question = input("\nAsk a question: ")

results = retrieve_chunks(question)

print("\nQuestion:", question)
print("\nRetrieved chunks:")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"\n{i}. {document}")
    print(f"Distance: {results['distances'][0][i - 1]:.4f}")

best_distance = results["distances"][0][0]

unsupported_keywords = [
    "tuition",
    "fee",
    "fees",
    "cost",
    "price",
]

if any(keyword in question.lower() for keyword in unsupported_keywords):
    print("\nGenerated Answer:")
    print("I don't have enough information in the knowledge base to answer that question.")
    raise SystemExit

context = "\n\n".join(
    f"- {document}" for document in results["documents"][0]
)

prompt = f"""
You are a university information assistant.

Answer the user's question using ONLY the information provided in the context below.

If the answer is not contained in the context, say:
"I don't have enough information in the knowledge base to answer that question."

Do not invent, assume, or add facts that are not present in the context.

Context:
{context}

Question:
{question}
"""

print("\nSending request to Gemini...")

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input=prompt,
    generation_config={
        "thinking_level": "low"
    },
)

print("Gemini response received.")

print("\nGenerated Answer:")
print(interaction.output_text)

