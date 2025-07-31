import pandas as pd 
from langchain_core.documents import Document 

class DataConverter:
    def __init__(self):
        print("DataConverter class has been initialized")
        self.product_data = pd.read_csv(r"data/flipkart_product_review.csv")
        pass 

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


if __name__ == '__main__':
    data_ingestion = DataConverter()
    data_ingestion.transform_data()