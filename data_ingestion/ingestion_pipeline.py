from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Tuple
from langchain_core.documents import Document
from utils.model_loader import ModelLoader 
from config.config_loader import load_config

class DataIngestion:
    def __init__(self):
        print("DataIngestion class has been initialized")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config = load_config()

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
        
    def _get_csv_path(self):
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, 'data', 'flipkart_product_review.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path
    
    def _load_csv(self):
        df = pd.read_csv(self.csv_path)

        expected_columns = {'review','product_id','product_title','rating','summary'}

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError("CSV must contain columns: {expected_columns}")

        return df
    
    def transform_data(self):
        product_docs = []
        for index, row in self.product_data.iterrows():
            product_docs.append(Document(page_content=row['review'], 
                metadata={
                    'product_id': row['product_id'],
                    'product_title': row['product_title'],
                    'rating': row['rating'],
                    'summary': row['summary']
                }
            ))

        return product_docs

    def store_in_vector_db(self, documents: List[Document]):
        collection_name = self.config["astra_db"]["collection_name"]
        vector_store = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace
        )

        inserted_ids = vector_store.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB")
        return vector_store, inserted_ids

    def run_pipeline(self):
        documents = self.transform_data()
        vector_store, inserted_ids = self.store_in_vector_db(documents)

        query = "Can you tell me the low budget headphone?"
        results = vector_store.similarity_search(query)

        for res in results:
            print(f"Content: {res.page_content} \n Metadata: {res.metadata}\n")
    
if __name__ == '__main__':
    data_ingestion = DataIngestion()
    data_ingestion.run_pipeline()