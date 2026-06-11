import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

VECTOR_STORE_DIR = "data/embeddings"
COLLECTION_NAME  = "faq_docs"

SUPPORTED_EXTENSIONS = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt":  TextLoader,
}


def load_documents(folder: str) -> list:
    """Carga todos los documentos soportados de una carpeta."""
    docs = []
    folder_path = Path(folder)

    for file_path in folder_path.rglob("*"):
        ext = file_path.suffix.lower()
        loader_class = SUPPORTED_EXTENSIONS.get(ext)

        if loader_class:
            print(f"  📄 Cargando: {file_path.name}")
            try:
                loader = loader_class(str(file_path))
                docs.extend(loader.load())
            except Exception as e:
                print(f"  ⚠️  Error en {file_path.name}: {e}")

    return docs


def ingest(folder: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Pipeline completo de ingesta:
    1. Cargar documentos
    2. Dividir en chunks
    3. Generar embeddings
    4. Almacenar en ChromaDB
    """
    print(f"\n🔍 Buscando documentos en: {folder}")
    docs = load_documents(folder)

    if not docs:
        print("❌ No se encontraron documentos soportados.")
        return

    print(f"✅ {len(docs)} documentos cargados. Dividiendo en chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "?", "!", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} chunks generados.")

    print("🧠 Generando embeddings y almacenando en ChromaDB...")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"✅ Vector store actualizado en: {VECTOR_STORE_DIR}")
    print(f"   Colección: '{COLLECTION_NAME}' — {len(chunks)} documentos indexados\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestar documentos para FAQ")
    parser.add_argument("--folder", default="data/documents/", help="Carpeta con documentos")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    ingest(args.folder, args.chunk_size, args.chunk_overlap)
