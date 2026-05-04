import os
import re
import sqlite3
import faiss
import numpy as np
from collections import deque

# --- STRICT OFFLINE ENVIRONMENT ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama 

class SyntberAI:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Absolute Paths
        self.db_path = os.path.join(self.base_dir, "syntber_knowledge.db")
        self.index_path = os.path.join(self.base_dir, "syntber.index")
        self.model_path = os.path.join(self.base_dir, "Phi-3-mini-4k-instruct-Q4_K_M.gguf")
        self.embed_path = os.path.join(self.base_dir, "models", "all-MiniLM-L6-v2")

        print("[LOG] Initializing Neural Core...")
        # local_files_only=True ensures no pings to HuggingFace
        self.embed_model = SentenceTransformer(self.embed_path, local_files_only=True, device="cpu")
        self.index = faiss.read_index(self.index_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Phi-3 configuration: 4 threads for your i5, 2048 context window
        self.llm = Llama(model_path=self.model_path, n_ctx=2048, n_threads=4, verbose=False)

        # Memory stores both user and AI turns to prevent "amnesia"
        self.memory = deque(maxlen=3) 
        
        # Performance Filters
        self.medical_triggers = {'bandage', 'wound', 'pill', 'doctor', 'treat', 'medicine', 'pain', 'cure', 'health', 'arm hurts'}
        self.noise = {'tell', 'show', 'give', 'about', 'know', 'find', 'what', 'is', 'how', 'the', 'a', 'of', 'to', 'me', 'please'}

    def search(self, query, top_k=3):
        """Optimized ID-based lookup for 150k facts"""
        q_vec = self.embed_model.encode([query]).astype('float32')
        distances, indices = self.index.search(q_vec, 10)
        
        query_words = [w for w in re.findall(r'\w+', query.lower()) if w not in self.noise and len(w) > 2]
        candidates = []
        cursor = self.conn.cursor()
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            # PERFORMANCE FIX: Using rowid lookup is significantly faster than OFFSET
            cursor.execute("SELECT answer FROM facts WHERE rowid=?", (int(idx) + 1,))
            row = cursor.fetchone()
            
            if row:
                text = row[0]
                dist = float(distances[0][i])
                matches = sum(1 for word in query_words if word in text.lower())
                
                # Dynamic scoring: weight keyword matches over simple vector distance
                score = (matches * 5.0) + (1.0 / (dist + 0.001))
                if query_words and matches == 0 and dist > 1.15: continue
                candidates.append((score, text))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:top_k]]

    def chat(self, user_input):
        input_lower = user_input.lower()
        
        # 1. HARD MEDICAL GUARDRAIL (Instant block)
        if any(t in input_lower for t in self.medical_triggers):
            return "I am a research AI. I am strictly prohibited from providing medical advice, first aid, or health assessments."

        # 2. VECTOR SEARCH (Find relevant facts)
        facts = self.search(user_input)
        
        # 3. DYNAMIC SYSTEM PROMPT (Solves Hallucination)
        if facts:
            context = "\n".join([f"- {f}" for f in facts])
            sys_msg = (
                "You are Syntber, a local AI researcher. Use the provided DATABASE FACTS. "
                "If the facts don't answer the question, answer using your general knowledge but be honest about your limits.\n\n"
                f"DATABASE FACTS:\n{context}"
            )
        else:
            sys_msg = (
                "You are Syntber. No matching database facts found. Answer based on general knowledge. "
                "If you don't know (e.g., very new hardware), admit you don't have that specific data."
            )

        # 4. CONSTRUCT PROMPT WITH MEMORY (Solves Amnesia)
        prompt = f"<|system|>\n{sys_msg}\n<|end|>\n"
        for past_u, past_a in self.memory:
            prompt += f"<|user|>\n{past_u}\n<|end|>\n<|assistant|>\n{past_a}\n<|end|>\n"
        prompt += f"<|user|>\n{user_input}\n<|end|>\n<|assistant|>\n"

        # 5. GENERATION (Low temperature for accuracy)
        output = self.llm(prompt, max_tokens=350, temperature=0.1, stop=["<|end|>"])
        answer = output['choices'][0]['text'].strip()
        
        # Update memory
        self.memory.append((user_input, answer))
        return answer

if __name__ == "__main__":
    ai = SyntberAI()
    print("\n--- SYNTBER PRO (STRICT OFFLINE) ---")
    while True:
        u = input("\nYou: ")
        if u.lower() in ['exit', 'quit']: break
        if not u.strip(): continue
        print(f"\nAI: {ai.chat(u)}")
