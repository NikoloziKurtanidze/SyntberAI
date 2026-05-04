import os
import sys
from sentence_transformers import SentenceTransformer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")

sys.stdout.write("--- SYNTBER MODEL BOOTSTRAP ---\n")
sys.stdout.write("[*] Downloading model weights from Hugging Face...\n")


model = SentenceTransformer('all-MiniLM-L6-v2')


if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)

sys.stdout.write(f"[*] Saving to: {MODEL_PATH}\n")
model.save(MODEL_PATH)

sys.stdout.write("\n[SUCCESS] Local brain initialized. You can now operate offline.\n")
sys.stdout.write("Next step: Run your indexer (faster.py or build_index.py).\n")
