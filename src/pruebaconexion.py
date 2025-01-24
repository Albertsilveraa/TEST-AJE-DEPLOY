#import os
#from qdrant_client import QdrantClient
#from sentence_transformers import SentenceTransformer
#import traceback

#QDRANT_HOST = os.getenv("QDRANT_HOST")
#QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
#COLLECTION_NAME = "drinks_collection"

# Inicializar cliente y modelo
#qdrant_client = QdrantClient(
 #   url=QDRANT_HOST,
  #  api_key=QDRANT_API_KEY
#)

#print(qdrant_client.get_collections())
# Probar la conexión
#try:
 #   response = qdrant_client.get_collections()
  ## print(response)
#except Exception as e:
 #   print(f"Error al conectar con Qdrant Cloud:")
  #  traceback.print_exc()
import chromadb

def test_chroma_connection():
    # Initialize ChromaDB client with persistent storage
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # List collections
    collections = chroma_client.list_collections()
    
    print("Available Collections:")
    for collection in collections:
        print(f"- {collection}")
        
    print(f"\nTotal Number of Collections: {len(collections)}")

# Run the connection test
test_chroma_connection()