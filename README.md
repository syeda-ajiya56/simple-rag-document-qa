# Simple RAG Document Q&A

A simple Retrieval-Augmented Generation (RAG) system that answers university-related questions using a small fictional knowledge base stored in a Chroma vector database.

## Project Overview

This project demonstrates the basic RAG pipeline:

1. A user's question is converted into an embedding.
2. The embedding is compared with information stored in Chroma.
3. Relevant knowledge chunks are retrieved.
4. The retrieved context is provided to a Gemini language model.
5. Gemini generates an answer using only the retrieved context.
6. If the knowledge base does not contain the requested information, the system returns a safe fallback instead of inventing an answer.

The application runs as a simple command-line interface (CLI). No web UI is required.

## Purpose

The goal of this project is to understand how Retrieval-Augmented Generation can connect a vector database with a language model.

RAG is useful when an AI system needs to answer questions using specific information such as:

* Company documentation
* University information
* Product documentation
* Customer support content
* Internal policies
* Research documents

## Technologies Used

* Python
* Google Gemini API
* `google-genai`
* ChromaDB
* `python-dotenv`
* Gemini Embedding Model: `gemini-embedding-001`
* Gemini Generation Model: `gemini-3.8-flash`

## RAG Architecture

```text
User Question
      |
      v
Generate Query Embedding
      |
      v
Chroma Vector Database
      |
      v
Retrieve Relevant Knowledge Chunks
      |
      v
Build Context + User Question
      |
      v
Gemini Language Model
      |
      v
Generated Answer
```

For unsupported questions:

```text
User Question
      |
      v
Retrieve Knowledge
      |
      v
Unsupported Information Detected
      |
      v
Safe Fallback Answer
```

## Knowledge Base

The project uses a fictional university knowledge base containing 10 information chunks.

The chunks cover topics including:

* Admissions
* Academic programs
* Financial aid
* Library services
* Accommodation
* Examinations
* Student life
* Career services
* Technical support
* Academic services

Each chunk contains:

* A unique ID
* Text content
* A category

The chunks are intentionally small and meaningful so that each piece represents a focused piece of information.

## Embeddings

Each knowledge chunk is converted into a numerical vector using Google's `gemini-embedding-001` model.

The embedding represents the semantic meaning of the text.

When a user asks a question, the question is also converted into an embedding. Chroma then searches for knowledge chunks that are semantically related to the question.

This allows the system to retrieve relevant information even when the user's wording is different from the wording in the knowledge base.

## Vector Database

ChromaDB is used as the local vector database.

The collection is named:

```text
university_knowledge
```

The database stores:

* Document text
* Embeddings
* Document IDs
* Category metadata

The project uses a persistent Chroma database stored locally in:

```text
chroma_db/
```

The `.gitignore` file excludes this directory from Git because it is generated local data.

## Retrieval

The system retrieves the top 3 relevant chunks for each question.

Chroma returns a distance value for each result. In this project, a lower distance represents a closer semantic match.

The retrieved documents are then combined into the context sent to the language model.

## Generation

The retrieved chunks and the user's question are included in a prompt instructing Gemini to:

* Answer using only the provided context.
* Avoid inventing information.
* State when the knowledge base does not contain enough information.

This helps reduce hallucination when the requested information is unavailable.

## Tested Questions

### Test 1 — Financial Aid

**Question:**

> Can students get help paying for their education?

**Top retrieved result:**

Scholarships

**Distance:**

`0.7049`

**Generated answer:**

> Yes. Students can apply for merit-based scholarships through the financial aid office, with decisions based on the university's published eligibility requirements.

**Result:** Successful.

---

### Test 2 — Undergraduate Application

**Question:**

> What do I need to submit when applying for an undergraduate program?

**Top retrieved result:**

Admissions

**Distance:**

`0.5123`

**Generated answer:**

> Based on the provided context, you need to complete the application form and submit the required academic documents before the published deadline.

**Result:** Successful.

---

### Test 3 — Internship Support

**Question:**

> Where can I get help preparing for an internship?

**Top retrieved result:**

Career Center

**Distance:**

`0.5214`

**Generated answer:**

> You can get help preparing for an internship at the university career center.

**Result:** Successful.

---

### Test 4 — Unsupported Information

**Question:**

> What is the tuition fee for the computer science program?

The retrieval step returned a computer science program chunk, but that chunk only described available programs and did not contain tuition fee information.

The top retrieved result had a distance of:

`0.6652`

Because the requested tuition/fee information is not available in the knowledge base, the system returned:

> I don't have enough information in the knowledge base to answer that question.

**Result:** Safe fallback successful.

The system did not invent a tuition fee.

## CLI Usage

Run the application from the project directory:

```powershell
python rag_qa.py
```

The application asks:

```text
Ask a question:
```

Enter a question, for example:

```text
Where can I get help preparing for an internship?
```

The system retrieves relevant knowledge and generates an answer when the required information is available.

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install chromadb google-genai python-dotenv
```

### 4. Create `.env`

Add your Gemini API key:

```env
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

Never commit the API key to Git.

### 5. Run the application

```powershell
python rag_qa.py
```

On the first run, the application generates embeddings and stores the knowledge base in Chroma.

On later runs, the existing Chroma collection is reused instead of generating the embeddings again.

## Security

The Gemini API key is stored in an environment variable rather than directly in the Python source code.

The following files and directories are excluded from Git:

```text
.env
.venv/
__pycache__/
chroma_db/
```

The API key should never be committed to a public repository.

## Limitations

This is a small educational RAG prototype rather than a production document-Q&A system.

Current limitations include:

* The knowledge base contains only 10 fictional chunks.
* The knowledge base is stored locally.
* Documents are manually created rather than uploaded dynamically.
* Retrieval is limited to the top 3 chunks.
* There is no authentication or user management.
* The application does not provide a web interface.
* The unsupported-question detection currently includes specific unsupported keywords.
* The quality of generated answers depends on retrieval quality and the available knowledge.
* The Gemini Free Tier has request limits.

## What This Project Demonstrates

This project demonstrates the complete basic RAG workflow:

**Retrieve → Augment → Generate**

It shows how embeddings can connect a user's natural-language question with semantically related information in a vector database, and how that retrieved information can then be supplied to a language model to generate a grounded response.

The unsupported-question test also demonstrates an important RAG principle: when the required information is not present in the available knowledge base, the system should avoid making up an answer.

## Evaluation and Guardrails

This project was extended with a basic evaluation process and a retrieval-based guardrail to make the RAG system more reliable when information is missing from the knowledge base.

### Why Evaluation Matters

A generative AI application can produce an answer even when the required information is not available. This can lead to hallucinations, where the model generates information that is unsupported by the available evidence.

Evaluation provides a repeatable way to test whether the application behaves as expected after changing the prompt, model, retrieval system, or guardrails.

### Guardrails Implemented

The evaluation version of the application uses two complementary guardrails.

#### 1. Retrieval-distance guardrail

The system checks the distance of the best retrieved knowledge chunk.

If the best retrieval distance is greater than `0.70`, the system does not send the question to Gemini and instead returns:

```text
I don't have enough information in the knowledge base to answer that question.
```

This prevents clearly unrelated questions from being passed to the language model.

#### 2. Context-only generation rule

When the retrieved context is sufficiently relevant, Gemini receives an explicit instruction to answer only from the retrieved context.

The system instructs the model not to:

* Invent information
* Assume missing details
* Estimate values that are not provided
* Add facts that are not present in the knowledge base

If the requested information is not fully supported by the context, the model is instructed to return the same safe fallback message.

### Evaluation Dataset

The evaluation contains 7 test questions covering three categories:

| Category | Tests | Purpose |
|---|---:|---|
| Supported | 3 | Verify that known information is answered correctly |
| Partially supported | 2 | Verify that missing details are not invented |
| Unsupported | 2 | Verify that completely unavailable information is rejected |
| **Total** | **7** | **Full evaluation set** |

The evaluation cases include questions about undergraduate programs, library resources, internship support, scholarship amounts, accommodation costs, tuition fees, and university rankings.

### Evaluation Results

The final evaluation produced the following result:

```text
Total tests: 7
Passed: 7
Failed: 0
Evaluation score: 100.0%
Overall result: PASS
```

The observed best retrieval distances were:

| Test type | Observed distance range |
|---|---:|
| Supported | 0.3422 – 0.5021 |
| Partially supported | 0.4202 – 0.5879 |
| Unsupported | 0.7877 – 0.9197 |

The two completely unsupported questions exceeded the `0.70` retrieval-distance threshold and were rejected by the retrieval guardrail.

The partially supported questions were semantically related to existing knowledge, but the requested details (scholarship amount and accommodation cost) were not present. The system correctly refused to invent those missing values.

### Before vs After

A baseline evaluation was also performed before adding the new retrieval-distance guardrail.

The baseline produced acceptable behavior for the five initial test questions. In particular, it did not invent tuition fees or a university ranking; it stated that the information was unavailable.

The improvement therefore does not claim that the baseline hallucinated on those specific tests. Instead, the guardrail makes the application's refusal behavior more explicit and enforceable by checking retrieval relevance before calling the language model and by restricting generation to retrieved context.

| Behavior | Before | After |
|---|---|---|
| Supported questions | Answered from retrieved context | Answered from retrieved context |
| Missing details | Model indicated information was unavailable in tested cases | Explicit safe fallback |
| Completely unsupported questions | Model indicated information was unavailable in tested cases | Retrieval guardrail blocks generation |
| Context restriction | Prompt-based | Retrieval check + context-only prompt |
| Repeatable evaluation | Manual tests | 7-case evaluation dataset |
| Final evaluation | Baseline observed | **7/7 PASS (100%)** |

### Running the Evaluation

The evaluation can be run with:

```powershell
python evaluate.py
```

The baseline behavior can be tested with:

```powershell
python baseline.py
```

The normal RAG application remains available through:

```powershell
python rag_qa.py
```

### Evaluation Limitations

This evaluation is intentionally small and educational.

The current evaluation has several limitations:

* Only 7 test questions are included.
* The knowledge base contains fictional university information.
* The `0.70` retrieval-distance threshold was selected based on the observed distance separation in this small dataset.
* Retrieval distance is not a perfect measure of whether an answer is factually supported.
* The evaluation checks expected behavior rather than performing a full semantic comparison of every generated answer.
* A larger production system would require a more diverse evaluation dataset, regression tests, human review, and additional safety checks.
* API rate limits can temporarily affect repeated evaluation runs when using a free-tier model.

### Files Added for Evaluation

The project now includes:

```text
rag_qa.py
baseline.py
evaluate.py
```

`rag_qa.py` is the original interactive RAG application.

`baseline.py` provides a simple before-guardrail comparison.

`evaluate.py` runs the repeatable 7-question evaluation and tests the new guardrail behavior.

### AI Work Disclosure

AI assistance was used during development to help reason about evaluation design, guardrail implementation, test cases, and documentation.

The final implementation was tested locally against the project's knowledge base, and the reported evaluation results are based on actual runs of the application.
