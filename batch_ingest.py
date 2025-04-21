import os
from PyPDF2 import PdfReader
from docx import Document
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Constants
DATA_FOLDER = "C:/Users/rashi/OneDrive/WISE/Knowledge"
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DIMENSIONS = 384
BATCH_SIZE = 100

# Connect to Zilliz
connections.connect(
    alias="default",
    uri="https://in03-357b70cf3851670.serverless.gcp-us-west1.cloud.zilliz.com",
    token="ce5060c7939d564fa7ae65d5c85cad6462b6b6fe5b0a8afc6216c7e3bd80da0aeb3ed4688157c2af9a36ddd30bc5838f9f53d880"
)

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(texts):
    return model.encode(texts).tolist()

def read_pdf_chunks(filepath):
    reader = PdfReader(filepath)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return [text[i:i+1000] for i in range(0, len(text), 1000)]

def read_docx_chunks(filepath):
    doc = Document(filepath)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [text[i:i+1000] for i in range(0, len(text), 1000)]

def extract_chunks(filepath):
    if filepath.endswith(".pdf"):
        return read_pdf_chunks(filepath)
    elif filepath.endswith(".docx"):
        return read_docx_chunks(filepath)
    return []

def get_collection(collection_name):
    if utility.has_collection(collection_name):
        return Collection(name=collection_name)
    else:
        raise ValueError(f"Collection '{collection_name}' does not exist.")

def ingest_file(filepath, collection_name):
    print(f"\U0001F4C4 Ingesting: {filepath} → {collection_name}")
    chunks = extract_chunks(filepath)
    if not chunks:
        print("⚠️ No chunks extracted.")
        return

    vectors = embed_texts(chunks)
    sources = [os.path.basename(filepath)] * len(chunks)
    ids = list(range(1, len(chunks) + 1))

    data = {
        "vector": vectors,
        "Text": chunks,
        "Source": sources
    }

    collection = get_collection(collection_name)
    collection.insert(data)
    print(f"✅ Inserted {len(chunks)} chunks.")

def scan_folder(base_path):
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.pdf', '.docx')):
                full_path = os.path.join(root, file)
                folder_name = os.path.basename(root).lower()
                collection_name = f"{folder_name}"
                try:
                    ingest_file(full_path, collection_name)
                except Exception as e:
                    print(f"❌ Failed to insert {file}: {e}")

if __name__ == "__main__":
    scan_folder(DATA_FOLDER)
