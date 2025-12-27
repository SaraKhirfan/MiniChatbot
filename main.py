"""
GenAI Study - Lab 8
A Retrieval-Augmented Generation (RAG) system designed to answer questions 
strictly based on Module 1: Foundations of Generative AI.
"""

import os
from langchain_openai import ChatOpenAI 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 

# ==========================================
# 1. INITIALIZE LLM (GENERATOR)
# ==========================================

# We use gpt-3.5-turbo with a low temperature (0.5) to ensure 
# answers remain focused and professional without excessive "creativity."
llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.5,
    openai_api_key="yourapikey" 
)

# ==========================================
# 2. KNOWLEDGE BASE (KNOWLEDGE SOURCE)
# ==========================================

# This list acts as our ground truth, extracted from Module 1 lecture notes.
course_material = [
    "AI encompasses strategies for making machines perform tasks that typically require human intelligence.",
    "Machine Learning (ML) uses existing data to train mathematical models to recognize patterns and make predictions.",
    "Supervised learning uses labeled input-output pairs for prediction or classification, while unsupervised learning finds patterns in unlabeled data.",
    "Deep learning uses deeply layered neural networks and excels at working with unstructured data and complex relationships.",
    "Discriminative models draw boundaries to classify data, while generative models learn underlying distributions to create new data.",
    "Generative AI (GenAI) is designed to create new content—such as text, images, and audio—based on patterns learned from existing data.",
    "Variational Autoencoders (VAEs) consist of an Encoder that maps input to a latent space and a Decoder that generates new samples.",
    "VAEs maximize the Evidence Lower Bound (ELBO) as a mathematically tractable approximation of the data log-likelihood.",
    "Generative Adversarial Networks (GANs) involve a two-player game between a Generator and a Discriminator to create realistic data.",
    "Transformers are the backbone of modern GenAI and rely on a highly parallel structure built on the Attention Mechanism.",
    "The Attention Mechanism weights token importance using Query (Q), Key (K), and Value (V) vectors.",
    "The Scaling Hypothesis focuses on simultaneously scaling model size, dataset size, and compute to increase model capacity.",
    "RAG combines the power of retrieval and generative models to answer queries accurately."
]

# ==========================================
# 3. VECTOR DATABASE (RETRIEVER)
# ==========================================

# HuggingFaceEmbeddings converts our text sentences into numerical vectors.
embeddings = HuggingFaceEmbeddings()

# FAISS (Facebook AI Similarity Search) stores these vectors for fast 
# semantic search when a user asks a question.
vector_store = FAISS.from_texts(course_material, embeddings)

# ==========================================
# 4. PROMPT ENGINEERING
# ==========================================

# This template enforces the "Strict Assistant" behavior.
# It prevents "Hallucinations" by limiting the LLM to the provided context.
prompt_template_str = """
You are a strict course assistant. 
Use ONLY the provided Course Material to answer the user's question. 

Rules:
1. If the answer is not in the material, say: "I'm sorry, that is not mentioned in my study notes."
2. If the user says "Hello" or "Hi", reply: "Hello! Please ask a question based on your course material."
3. Do not use any outside knowledge.

Course Material:
{course_material}

Question: {question}
Answer:
"""

prompt = PromptTemplate.from_template(prompt_template_str)

# ==========================================
# 5. CORE RAG LOGIC
# ==========================================

def get_answer(question):
    """
    Processes a user question through a RAG pipeline.
    
    Args:
        question (str): The student's inquiry.
        
    Returns:
        str: The AI-generated answer based solely on course material.
    """
    
    # PHASE 1: Retrieval
    # Search the FAISS database for the most relevant sentence (k=5).
    docs = vector_store.similarity_search(question, k=5)
    context = docs[0].page_content

    # PHASE 2: Generation
    # Create a LCEL (LangChain Expression Language) chain:
    # 1. Format the prompt with context + question
    # 2. Pass to GPT-3.5
    # 3. Clean the output into a string
    chain = prompt | llm | StrOutputParser()
    
    # Execute the chain and return the result
    answer = chain.invoke({
        "course_material": context, 
        "question": question
    })
    
    return answer