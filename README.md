# 🧱RAG agent for PDF-based QA task
## 🧠 Use case
This project demonstrates how to build a **RAG-based QA agent** for automating the manual effort of reading research papers.  
By combining a **domain-specific LLM** with an **embedding model**, and leveraging **context retrieved through similarity search**, the agent can:  

- Answer **domain-specific questions** directly from research papers in PDF format
- Go beyond general-purpose QA to provide **research-focused insights** 
- Improve efficiency by reducing the need for manual paper review  


## 🚀 Getting Started

### 📦 1. Create a New Environment
Create a new environment and install all dependencies through the provided `.txt` file.
```bash
conda create -n myenv -y
conda activate myenv
pip install -r requirements.txt
```
### 📦 2. Clone the repo
To start working on this project locally, clone this repository:
 ```bash
git clone https://github.com/Subhigupta/RAG-based-Intelligent-QA.git
```

## 📁 Project Structure Guide

This repository is organized into the following key directories:

### `research_docs/`
Contains two research documents that discuss data-driven, knowledge-based, and hybrid modeling techniques used in the biopharma industry, along with highlighting their respective advantages and disadvantages.

### `prototype/`
This folder contains interactive Jupyter notebooks created during the early development phase of the project.  
They demonstrate experiments with different models and provide insights into their performance for the QA task.
- **Embedding Models**  
  - `sentence-transformers/all-mpnet-base-v2`  
  - `NeuML/pubmedbert-base-embeddings`  

- **LLM Models**  
  - `google/flan-t5-large`  
  - `TheBloke/PMC_LLAMA-7B-GPTQ`

Exploring these notebooks will help you understand the modeling workflow and identify which models performed best for the QA task.

### `app.py`
An automated pipeline script that integrates all steps required to build a RAG agent.
It allows the user to interactively query the system multiple times until they choose to exit.