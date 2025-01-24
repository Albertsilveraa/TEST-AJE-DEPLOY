import pandas as pd
import re
import os
import uuid
import openai
import chromadb
from tqdm import tqdm
from src.limpiezadatos import *

# Configure OpenAI API key
openai.api_key = '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'  # Replace with your actual API key

# Embedding Generation Function
def generate_embedding(text):
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        print(f"Error generating embedding for text: {text}\nError: {e}")
        return None

# Prepare text for embedding
combined_df['text_for_embedding'] = (
    combined_df['name'] + " " + 
    combined_df['sub_category'] + " " + 
    "Rating:" + combined_df['ratings'].astype(str) + " " + 
    "Price:" + combined_df['actual_price'].astype(str)
)
# Generate embeddings
embeddings = []
texts = []

print("Generating embeddings...")
for text in tqdm(combined_df['text_for_embedding']):
    embedding = generate_embedding(text)
    if embedding:
        embeddings.append(embedding)
        texts.append(text)

# Initialize ChromaDB client (updated configuration)
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Create or get collection
collection = chroma_client.get_or_create_collection(name="Drinks-collection")

# Prepare data for insertion
ids = combined_df['id'].tolist()
metadatas = combined_df.drop(columns=['text_for_embedding']).to_dict(orient='records')

# Insert embeddings
print("Inserting embeddings into Chroma DB...")
collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print("Embeddings generated and stored in Chroma DB successfully.")