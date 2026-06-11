import os
from dotenv import load_dotenv

load_dotenv()

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

db = Chroma(
    persist_directory="data/embeddings",
    embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
    collection_name="faq_docs",
)

print(f"Total de documentos indexados: {db._collection.count()}")

queries = ["horario de atención", "requisitos de matrícula", "transporte escolar"]

print("\n--- Resultados de búsqueda ---")
for query in queries:
    print(f"\n🔍 Query: '{query}'")
    results = db.similarity_search(query, k=2)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.page_content[:100]}...")
