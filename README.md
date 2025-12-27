# 🎓 GenAI Study 

(Lab 8 - Gen AI Internship - HTU - Dec 2025),
an intelligent, Retrieval-Augmented Generation (RAG) assistant designed to help students master the foundations of Generative AI based on specific course materials.

---

## 📌 Project Overview
This project implements a **RAG (Retrieval-Augmented Generation) pipeline** to provide a "Strict Course Assistant." Unlike standard AI, this assistant is restricted to a specific knowledge base (Module 1), ensuring that answers are accurate, grounded in lecture notes, and free from external hallucinations.

## 🏗️ Architecture
The system follows a modern AI stack:
1.  **Data Source**: Expert-curated notes from "Foundations of Generative AI."
2.  **Embeddings**: `HuggingFaceEmbeddings` used to transform text into high-dimensional vectors.
3.  **Vector Store**: `FAISS` (Facebook AI Similarity Search) for efficient semantic retrieval.
4.  **Generator**: `GPT-3.5-Turbo` via LangChain to synthesize answers from retrieved context.



---

## 🚀 Key Features
* **Context-Aware Retrieval**: Uses semantic search to find the most relevant study notes for any query.
* **Strict Guardrails**: Programmed to refuse answering questions outside the scope of the provided material.
* **Glassmorphism UI**: A modern, responsive web interface with an animated gradient background and smooth loading transitions.
* **One-Time Preloader**: A professional entry animation that only triggers on initial site entry, not during every chat interaction.

---

## 🛠️ Technical Implementation

### The RAG Pipeline (Python)
The core logic utilizes **LangChain Expression Language (LCEL)** to chain the prompt and LLM:

python, the logic flow:
 1. User asks a question
 2. Similarity search finds relevant 'docs' in FAISS
 3. Prompt is filled with the retrieved 'context'
 4. LLM generates answer based ONLY on that context
chain = prompt | llm | StrOutputParser()
The Frontend (HTML5/CSS3/JS)
CSS Animations: Custom @keyframes for the moving gradient background and spinner.
JavaScript: Intelligent preloader management using the performance.getEntriesByType API to ensure the loading screen only appears on fresh visits, not on form submissions.

---

## 📝 Short Reflection
**What worked well?** The integration between FAISS and LangChain was seamless. The system's ability to retrieve exact citations from the text proved that the semantic search was highly accurate.

**What limitations did we face?** Standardizing text chunks was a challenge; if a chunk is too short, it lacks context, but if it is too long, it can exceed the LLM's prompt limit. We balanced this by using k=5 in our similarity search.

**How did RAG improve the answers?** RAG transformed a general-purpose model into a specialized tutor. It eliminated "hallucinations" by providing a reference source, ensuring that the assistant used course-specific terminology exactly as taught in the module.

---

## 🔧 Installation & Setup

Clone the repo:
`git clone [repository-url]`

Install dependencies:
`pip install langchain langchain-openai faiss-cpu sentence-transformers flask`

Set OpenAI Key:
Ensure your OPENAI_API_KEY is set in your environment variables.

Run the application:
`python app.py`
