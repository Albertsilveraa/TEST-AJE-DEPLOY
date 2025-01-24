import chromadb
import openai
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from settings import *
import os 
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# Clase principal del sistema de búsqueda
class VectorSearchSystem:
    def __init__(self, db_path="chroma_db", collection_name="Drinks-collection"):
        # Inicializar cliente de ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)
        
        # Configuración de OpenAI
        openai.api_key = "sk-proj-VXMAN26EjnlYjjlRjVfgN2EQONa7aFC28dYlq_wQ1XTiPgwOw37beXoRicSWgQ_NzQkIWib5_ET3BlbkFJ5DwOSmlch8d54JeUNcHdmJ_KoAbhZOr8LVeeR1kLkq4pxNla"
    def generate_embedding(self, text):
        """Genera un embedding para un texto dado usando OpenAI"""
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']

    def refine_query(self, query):
        """Refina la consulta del usuario usando el modelo de lenguaje"""
        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente especializado en mejorar consultas de búsqueda para productos relacionados "
                    "con bebidas. Tu objetivo es tomar consultas originales y convertirlas en versiones más claras, "
                    "detalladas y específicas para optimizar la búsqueda semántica.\n\n"
                    "- Mantén el significado original de la consulta, pero elimina cualquier ambigüedad.\n"
                    "- Si la consulta es demasiado genérica, añade detalles que puedan ayudar, como ejemplos de categorías, tamaños o tipos de bebidas si es relevante.\n"
                    "- No alteres términos técnicos o nombres de productos específicos."
                )
            },
            {"role": "user", "content": f"Consulta original: \"{query}\"\n\nConsulta refinada:"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        refined_query = response.choices[0].message['content'].strip()
        return refined_query

    def generate_natural_response(self, results, query):
        """Genera una respuesta en lenguaje natural basada en los resultados de la búsqueda"""
        if not results:
            return "No se encontraron resultados para tu búsqueda."

        formatted_results = "\n".join(
            [f"{i+1}. {result['name']} - Subcategoría: {result['sub_category']}, "
             f"Calificaciones: {result['ratings']}, Precio: {result['actual_price']}" 
             for i, result in enumerate(results)]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en productos de bebidas. Tu tarea es generar respuestas en lenguaje natural "
                    "que sean claras, útiles y relevantes para el usuario, basándote en los resultados de búsqueda.\n\n"
                    "- Usa un tono amigable y profesional.\n"
                    "- Si hay múltiples resultados, prioriza los más relevantes según las categorías o calificaciones.\n"
                    "- Si no hay resultados, proporciona una explicación y sugiere cómo reformular la consulta.\n"
                    "- Incluye detalles como el nombre del producto, subcategorías y calificaciones en la respuesta."
                )
            },
            {"role": "user", "content": f"Consulta del usuario: \"{query}\"\n\nResultados de búsqueda:\n{formatted_results}\n\nRespuesta:"}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        natural_response = response.choices[0].message['content'].strip()
        return natural_response

    def process_natural_language_query(self, query):
        """Refina la consulta y genera un embedding para la consulta refinada"""
        refined_query = self.refine_query(query)
        query_embedding = self.generate_embedding(refined_query)
        return query_embedding, refined_query

    def semantic_search(self, query, top_k=5):
        """Realiza una búsqueda semántica en la base de datos vectorial"""
        query_embedding, refined_query = self.process_natural_language_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['metadatas', 'distances']
        )
        
        processed_results = [
            {
                'name': results['metadatas'][0][i].get('name', 'N/A'),
                'sub_category': results['metadatas'][0][i].get('sub_category', 'N/A'),
                'ratings': results['metadatas'][0][i].get('ratings', 'N/A'),
                'actual_price': results['metadatas'][0][i].get('actual_price', 'N/A'),

                'distance': results['distances'][0][i]
            } for i in range(len(results['metadatas'][0]))
        ]
        
        return self.generate_natural_response(processed_results, refined_query)

    def advanced_filtering(self, query, filters=None, top_k=5):
        """Búsqueda avanzada con filtrado opcional"""
        query_embedding, refined_query = self.process_natural_language_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            where=filters or {},
            n_results=top_k,
            include=['metadatas', 'distances']
        )
        
        processed_results = [
            {
                'name': results['metadatas'][0][i].get('name', 'N/A'),
                'sub_category': results['metadatas'][0][i].get('sub_category', 'N/A'),
                'ratings': results['metadatas'][0][i].get('ratings', 'N/A'),
                'actual_price': results['metadatas'][0][i].get('actual_price', 'N/A'),
                'distance': results['distances'][0][i]
            } for i in range(len(results['metadatas'][0]))
        ]
        
        return self.generate_natural_response(processed_results, refined_query)

    def visualize_similarities(self, embeddings, labels):
        """Visualiza las similitudes entre productos usando PCA"""
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)

        plt.figure(figsize=(10, 6))
        for i, label in enumerate(labels):
            plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label=label)
            plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label)

        plt.title("Visualización de Similitudes entre Productos")
        plt.legend()
        plt.show()

# FastAPI para interacción con el sistema
app = FastAPI()

class SearchQuery(BaseModel):
    query: str
    filters: dict = None

search_system = VectorSearchSystem()

@app.post("/search")
def search(query: SearchQuery):
    try:
        response = search_system.semantic_search(query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/advanced_search")
def advanced_search(query: SearchQuery):
    try:
        response = search_system.advanced_filtering(query.query, filters=query.filters)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
