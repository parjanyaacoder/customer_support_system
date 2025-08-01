from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from typing import List
from utils.model_loader import ModelLoader 
from config.config_loader import load_config

class Retriever:

    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vector_store = None
        self.retriever = None
        pass 

    def _load_env_variables(self):
        load_dotenv()
        required_vars = ["ASTRA_DB_API_ENDPOINT","ASTRA_DB_APPLICATION_TOKEN","ASTRA_DB_KEYSPACE","GOOGLE_API_KEY"]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")
        self.db_api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace=os.getenv("ASTRA_DB_KEYSPACE")
        self.google_api_key=os.getenv("GOOGLE_API_KEY")
    
    def load_retriever(self):
        if not self.vector_store:
            collection_name = self.config['astra_db']['collection_name']
            self.vector_store = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace
        )

        if not self.retriever:
            top_k = self.config['retriever']['top_k'] if "retriever" in self.config else 3
            retriever = self.vector_store.as_retriever(search_kwargs={'k': top_k})
            print("Retriever loaded successfully")
            return retriever

    def call_retreiver(self, query: str) -> List[Document]:
        retriever = self.load_retriever()
        return retriever.invoke(query)

if __name__ == "__main__":
    retriever_obj = Retriever()
    user_query = "Can you suggest some good budget laptops ?"
    results = retriever_obj.call_retreiver(user_query)

    for idx, doc in enumerate(results, 1):
        print(f"Result {idx}: {doc.page_content} \n Metadata: {doc.metadata} \n")
