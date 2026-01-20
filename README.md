# RAG Educational System  
**A Retrieval-Augmented Generation System for Real-Time Educational Question Answering**

## Project Overview
This project presents the design and implementation of a **Retrieval-Augmented Generation (RAG)** system aimed at answering educational questions by combining:

- neural information retrieval (FAISS),
- dense text embeddings,
- large language models (LLMs),
- and real-time external data sources.

The system was developed as part of a **Bachelor’s thesis** and demonstrates how modern LLMs can be enhanced by grounding their responses in dynamically retrieved, relevant context.

## Project Objectives
- Design and implement a modular RAG architecture
- Use a FAISS-based neural retriever for semantic search
- Integrate both local and cloud-based LLMs
- Retrieve information from real-time external sources
- Separate system components into reusable, well-defined modules
- Provide a user-friendly web interface for interaction

## Data Sources
The system combines static and real-time information sources:

- **Google Custom Search** (primary live data source)
- **Wikipedia API**
- **Stack Exchange API**
- **OpenAlex API**
- **Local PDF documents**

All web-based sources are accessed using an **API-first approach**, without traditional web scraping.

## Language Models
The system supports two execution modes:

### Local Mode
- Ollama
- Qwen 2.5 Instruct (3B)

### Cloud Mode
- Hugging Face Inference API / Groq API
- Mistral / LLaMA family models

Switching between local and cloud execution is handled via environment variables.

## ⚙️ Technologies Used
- **Python**
- **FAISS**
- **Ollama**
- **Hugging Face / Groq APIs**
- **Streamlit**
- **Google Custom Search API**
- **Sentence Transformers (embedding models)**



