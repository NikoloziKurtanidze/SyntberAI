import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_fast_index():
    print("[LOG] Loading Embedding Model...")
    embed_model = SentenceTransformer("models/all-MiniLM-L6-v2", local_files_only=True, device="cpu")
    
    print("[LOG] Connecting to SQLite Database...")
    conn = sqlite3.connect("syntber_knowledge.db")
    cursor = conn.cursor()
    cursor.execute("SELECT rowid, answer FROM facts")
    rows = cursor.fetchall()
    
    if not rows:
        print("[ERROR] No data found in the facts table!")
        return

    print(f"[LOG] Encoding {len(rows)} facts. This will take a few minutes on an i5...")
    texts = [row[1] for row in rows]
    vectors = embed_model.encode(texts, show_progress_bar=True).astype('float32')


    vector_dimension = vectors.shape[1] 
    nlist = 1000
    
    quantizer = faiss.IndexFlatL2(vector_dimension)
    index = faiss.IndexIVFFlat(quantizer, vector_dimension, nlist, faiss.METRIC_L2)
    
    print("[LOG] Training the FAISS clusters...")
    index.train(vectors)
    
    print("[LOG] Adding vectors to clusters...")
    index.add(vectors)

    faiss.write_index(index, "syntber_fast.index")
    print("[SUCCESS] Fast Index built! Update your main script to use 'syntber_fast.index'.")

if __name__ == "__main__":
    build_fast_index()
