import os
import torch
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.vectorstores import FAISS
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_mongodb import MongoDBAtlasVectorSearch

load_dotenv()

class VectorDb():

    def __init__(self, user_db, chunks, embedding):

        self.user_db = user_db
        self.chunks = chunks
        self.embedding = embedding
        print(f"{user_db} vector database is getting set up...")

        if self.user_db == "AstraDB":
            vstore = self.create_astradb_store()

        if self.user_db == "FAISS":
            vstore = self.create_faiss_store()

        if self.user_db == "MongoDB":
            vstore = self.create_mongodb_store()

        self.vstore = vstore

    def create_astradb_store(self):
        # Load your API secret keys
        try:
            ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
            ASTRA_DB_ID = os.environ["ASTRA_DB_ID"]
            ASTRA_DB_API_ENDPOINT = os.environ["ASTRA_DB_API_ENDPOINT"]
            
        except:
            print("Mention your Astra db API keys in a .env file")

        # Convert extract chunks into embeddings and store them in vector database
        vstore = AstraDBVectorStore(embedding = self.embedding,
                                    collection_name = "langchain_pdf_query",
                                    api_endpoint = ASTRA_DB_API_ENDPOINT,
                                    token = ASTRA_DB_APPLICATION_TOKEN)

        vstore.add_documents(self.chunks)
        astra_vector_index = VectorStoreIndexWrapper(vectorstore = vstore)
    
        return vstore
    
    def create_faiss_store(self):

        if os.path.exists("faiss_index/"):
            print("Vector Database already exist locally...")
            vstore = FAISS.load_local("faiss_index", self.embedding, allow_dangerous_deserialization=True)
        else:
            vstore = FAISS.from_documents(self.chunks, self.embedding)
            vstore.save_local("faiss_index")
        
        return vstore

    def create_mongodb_store(self):

        #try:
        MONGODB_URI = os.environ["MONGODB_URI"]

        # Set the MongoDB URI, DB, Collection Names
        client = MongoClient(MONGODB_URI)
        dbName = "hybridModel_mongodb_chunks"
        collectionName = "chunked_data"
        collection = client[dbName][collectionName]
        ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

        vstore = MongoDBAtlasVectorSearch.from_documents(documents = self.chunks,
                                                                embedding = self.embedding,
                                                                collection = collection,
                                                                index_name = ATLAS_VECTOR_SEARCH_INDEX_NAME)
        
        return vstore
        # except:
        #     print("Mention your MongoDB API keys in a .env file")


        
        
        