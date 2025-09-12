# Import all libraries
import warnings
import logging
import shutil
import numpy as np
from src.data_ingestion import *
from src.vector_store import *
from src.rag_pipeline import *
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress all warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)  # Disable ALL logging below CRITICAL
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Make Hub downloads resilient on slower links
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "180"
os.environ["HF_HUB_DOWNLOAD_RETRY"]   = "20"

# Define embedding model
embedding = HuggingFaceEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")

print("Starting with setting up everything for you...")
# Step 1 : Load and preprocess the data
all_docs = read_pdfs("data/")
chunks= generate_chunks(all_docs, embedding)

# Step 2: Create vector database and store chunks of data
user_db = input("Specify your database from AStraDB, MongoDB, FAISS: ").strip() # ask user to specify database
vector_database = VectorDb(user_db, chunks, embedding)
vstore = vector_database.vstore

# Step 3: Run RAG pipeline (ask a question)
model_id  = "TheBloke/PMC_LLAMA-7B-GPTQ"
print("Loading the tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

os.makedirs("./model_offload", exist_ok=True)
print("Loading the model...")
model = AutoGPTQForCausalLM.from_quantized(model_id,
                                            device_map="auto",
                                            max_memory={0: "5GB", "cpu": "14GB"},  # Adjust based on your system
                                           offload_folder="./model_offload", use_safetensors=True, trust_remote_code=True)
print("-" * 50) 
print("RAG Question Answering System is ready...")
print("Type 'quit', 'exit', or 'stop' to end the session")
print("-" * 50)

while True:
    query = input("\nAsk your query: ").strip()

    # Check for exit commands
    if query.lower() in ['quit', 'exit', 'stop', 'q']:
        print("Thank you for using the RAG system. Goodbye!")
        break

    # Skip empty queries
    if not query:
        print("Please enter a valid query.")
        continue

    # Process the query
    answer = run_rag_pipeline(vstore, tokenizer, model, query)
    print(f"\nAnswer: {answer}")

if user_db=="AstraDB":
    vstore.clear()
elif user_db=="MongoDB":
    vstore.delete(ids=None)
else:
    shutil.rmtree("faiss_index")   # removes folder + all its contents
