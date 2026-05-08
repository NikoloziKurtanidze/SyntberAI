import os
import sys
import re
import math
import sqlite3
import multiprocessing
import shutil
import faiss
import numpy as np
from collections import deque


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_DIR, "cache")
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "cache")

try:
    from sentence_transformers import SentenceTransformer
    from llama_cpp import Llama
except ImportError:
    print("❌ Error: Missing dependencies. Run: pip install llama-cpp-python sentence-transformers faiss-cpu numpy")
    sys.exit(1)

class SyntberAI:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.models_dir = os.path.join(self.base_dir, "models")
        self.embed_path = os.path.join(self.models_dir, "all-MiniLM-L6-v2")
        self.db_path = os.path.join(self.base_dir, "syntber_knowledge.db")
        self.index_path = os.path.join(self.base_dir, "syntber_fast.index")
        self.llm_path = os.path.join(self.base_dir, "Phi-3-mini-4k-instruct-Q4_K_M.gguf")

        self.ai_name = "Syntber"
        self.creator_name = "The Syntber Project"


        self._check_and_fix_search_engine()
        self._verify_heavy_assets()

 
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


        cpu_threads = max(1, multiprocessing.cpu_count() - 1)
        
        sys.stdout.write("[*] Initializing Search Brain...\n")

        self.embed_model = SentenceTransformer(self.embed_path, local_files_only=True, device="cpu")
        
        self.index = faiss.read_index(self.index_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        sys.stdout.write(f"[*] Loading LLM Brain ({cpu_threads} threads)...\n")
        self.llm = Llama(
            model_path=self.llm_path, 
            n_ctx=2048, 
            n_threads=cpu_threads,
            n_gpu_layers=0,
            verbose=False
        )

        self.memory = deque(maxlen=2)
        self.medical_pattern = re.compile(r'\b(diagnose|surgery|medication|doctor|symptom|injury)\b', re.IGNORECASE)

    def _check_and_fix_search_engine(self):
        """Ensures the search engine is correctly dimensioned (384-bit)."""
        is_corrupted = False
        if os.path.exists(self.embed_path):

            if not os.path.exists(os.path.join(self.embed_path, "1_Pooling")):
                is_corrupted = True
        
        if not os.path.exists(self.embed_path) or is_corrupted:
            if is_corrupted:
                print("[!] Architecture mismatch detected. Cleaning corrupted files...")
                shutil.rmtree(self.embed_path)
            
            print("[*] Downloading search engine weights (First-time setup)...")
            os.makedirs(self.embed_path, exist_ok=True)
            temp_model = SentenceTransformer('all-MiniLM-L6-v2')
            temp_model.save(self.embed_path)
            print("[*] Search engine localized successfully.")

    def _verify_heavy_assets(self):
        """Checks for the GGUF, Index, and DB files."""
        missing = [p for p in [self.db_path, self.index_path, self.llm_path] if not os.path.exists(p)]
        if missing:
            print(f"\n❌ CRITICAL ERROR: Missing assets in root folder: {missing}")
            print("Please ensure your Google Drive files are next to this script.")
            sys.exit(1)

    def search(self, query):
        q_vec = self.embed_model.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(q_vec, 3)
        
        candidates = []
        cursor = self.conn.cursor()
        for p, idx in enumerate(indices[0]):
            if idx == -1 or distances[0][p] > 1.2: continue
            cursor.execute("SELECT answer FROM facts WHERE rowid=?", (int(idx) + 1,))
            row = cursor.fetchone()
            if row:
                candidates.append(row[0])
        return candidates[:2]

    def chat(self, user_input):
        input_cleaned = user_input.lower().strip()
        if self.medical_pattern.search(input_cleaned):
            return "[Safety Protocol]: Medical evaluation restricted."

        facts = self.search(user_input) if len(input_cleaned) > 3 else []
        
        sys_msg = (
            f"You are {self.ai_name}, a logic-driven AI. "
            "Use ONLY plain text for math. No LaTeX. "
            "Prioritize user variables like g=10."
        )
        if facts:
             sys_msg += "\n\nRelevant Knowledge:\n" + "\n".join([f"- {f}" for f in facts])

        prompt = f"<|system|>\n{sys_msg}<|end|>\n"
        for u_old, a_old in self.memory:
            prompt += f"<|user|>\n{u_old}<|end|>\n<|assistant|>\n{a_old}<|end|>\n"
        prompt += f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

        output = self.llm(prompt, max_tokens=256, temperature=0.0, stop=["<|end|>", "<|user|>"])
        answer = output['choices'][0]['text'].strip()
        self.memory.append((user_input, answer))
        return answer

if __name__ == "__main__":
    print("\n[*] Booting Syntber AI Engine...")
    ai = SyntberAI()
    

    print(f"\n{'='*50}")
    print(f" {ai.ai_name.upper()} V1.2.2 - LOCAL & PRIVATE ")
    print(f"{'='*50}")
    print("DISCLAIMER: Syntber is an AI and can be wrong.")
    print(" Always verify mathematical and scientific outputs.")
    print(" This software is NOT a substitute for professional")
    print(" medical, legal, or engineering advice.")
    print(f"{'='*50}\n")
    
    while True:
        try:
            u = input("You: ").strip()
            if u.lower() in ['exit', 'quit']: break
            if u:
                print(f"\nSyntber: {ai.chat(u)}\n")
        except KeyboardInterrupt:
            break
