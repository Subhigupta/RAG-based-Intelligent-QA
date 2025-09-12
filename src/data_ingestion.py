import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

# Function to read the contents of PDFs
def read_pdfs(dataset_path):

    print("Extracting PDF's...")

    all_docs = []

    for file in os.listdir(dataset_path):
        if file.endswith('.pdf'): 

            file_path = os.path.join(dataset_path, file)
            loader = PyPDFLoader(file_path, mode="single")
            docs = loader.load()

            if len(docs[0].page_content.split(" ")) > 20: #avoiding storing empty pages 
                all_docs.append(docs[0])
            
    return all_docs

# Function to divide the extracted text into chunks
def generate_chunks(all_docs, embedding):

    print("Chunks are being created...")
    
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 100,
                                                   #length_function = len)
    text_splitter = SemanticChunker(embedding, breakpoint_threshold_type="percentile", min_chunk_size=100, breakpoint_threshold_amount=80.0)
    chunks = text_splitter.split_documents(all_docs)
    
    return chunks