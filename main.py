import os
import sys
import re
import math
import sqlite3
import faiss
import numpy as np
from collections import deque

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    from sentence_transformers import SentenceTransformer
    from llama_cpp import Llama
except ImportError:
    sys.exit(1)

class SyntherAI:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_dir, "syntber_knowledge.db")
        self.index_path = os.path.join(self.base_dir, "syntber_fast.index")
        self.llm_path = os.path.join(self.base_dir, "Phi-3-mini-4k-instruct-Q4_K_M.gguf")
        self.embed_path = os.path.join(self.base_dir, "models", "all-MiniLM-L6-v2")

        self.ai_name = "Syntber"
        self.creator_name = "The Syntber Project"

        self._verify_assets()

        self.embed_model = SentenceTransformer(self.embed_path, local_files_only=True, device="cpu")
        self.index = faiss.read_index(self.index_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        self.llm = Llama(
            model_path=self.llm_path, 
            n_ctx=2048, 
            n_threads=4, 
            n_gpu_layers=0, 
            verbose=False
        )

        self.memory = deque(maxlen=2)
        self.medical_pattern = re.compile(
            r'\b(bandage|wound|diagnose|prescription|dosage|surgery|medication|health|doctor|hurt|pain|ache|symptom|injury|sore)\b', 
            re.IGNORECASE
        )
        self.small_talk = {'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'yo'}

        self.decay_rate_a = 1.3
        self.m1 = 1.0
        self.m2 = 1.0

    def _verify_assets(self):
        missing = [p for p in [self.db_path, self.index_path, self.llm_path, self.embed_path] if not os.path.exists(p)]
        if missing:
            sys.exit(1)

    def _exp_decay_pe(self, p, a, b, c, m1, m2, V):
        return min((math.pi ** (-a * b * p)) * c * m1 * m2 * V, 2.5)

    def search(self, query, c_factor):
        if c_factor <= 0.0:
            return []
        
        q_vec = self.embed_model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(q_vec, 5)
        
        candidates = []
        cursor = self.conn.cursor()
        
        for p, idx in enumerate(indices[0]):
            if idx == -1 or distances[0][p] > 1.2:
                continue
                
            cursor.execute("SELECT answer FROM facts WHERE rowid=?", (int(idx) + 1,))
            row = cursor.fetchone()
            
            if row:
                v_score = 1.0 / (float(distances[0][p]) + 0.001)
                score = self._exp_decay_pe(p, self.decay_rate_a, 5, c_factor, self.m1, self.m2, v_score)
                candidates.append((score, row[0]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:2]]

    def chat(self, user_input):
        input_cleaned = user_input.lower().strip()
        
        if self.medical_pattern.search(input_cleaned):
            return f"{self.ai_name} Safety Protocol: Medical evaluation restricted."

        c_factor = 0.0 if input_cleaned in self.small_talk or len(input_cleaned) < 3 else 1.0
        facts = self.search(user_input, c_factor)
        
        sys_msg = (
            f"You are {self.ai_name}, created by {self.creator_name}. "
            "The weights you use were built by Microsoft, but your architecture and logic belong to your creator. "
            "Respond concisely. Do NOT provide unsolicited math examples or tutorials. "
            "STRICT MATH RULE: Use user-provided variables (e.g., g=10) for calculations ONLY when a question is asked. "
            "Perform all math step-by-step."
            "Do not give medical information and advice strictly deny it"
            "FORMATTING RULE: Use ONLY plain text for math. Do NOT use LaTeX, brackets like [ ], "
            "or backslashes. Use simple symbols like * for multiply, / for divide, and ^2 for squares. "
            "Example: d = 0.5 * g * t^2"
            "Finish the sentences. "
            "You should asist users with both math-related and non-math-related topics that you are not forbiden from speaking. "
            "Do not write your thought process for example:(Note: Since no specific question was asked that requires mathematical assistance, the response remains general.) unless you are asked to."
        )
        
        if facts:
             sys_msg += "\n\nCONTEXT FROM YOUR KNOWLEDGE BASE:\n" + "\n".join([f"- {f}" for f in facts])

        prompt = f"<|system|>\n{sys_msg}<|end|>\n"
        for past_u, past_a in self.memory:
            prompt += f"<|user|>\n{past_u}<|end|>\n<|assistant|>\n{past_a}<|end|>\n"
        prompt += f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

        output = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.0,
            repeat_penalty=1.15,
            stop=["<|end|>", "<|user|>", "<|assistant|>", "###"]
        )
        
        answer = output['choices'][0]['text'].strip()
        self.memory.append((user_input, answer))
        return answer

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    ai = SyntherAI()
    print(f"\n{'='*40}")
    print(f" {ai.ai_name.upper()} V1.2.2: ONLINE ")
    print(f"{'='*40}")
    print(" DISCLAIMER: Syntber is an AI and can provide incorrect information.")
    print(" Always verify mathematical and factual outputs.")
    print(f"{'='*40}\n")
    
    while True:
        try:
            u = input("\nYou: ").strip()
            if u.lower() in ['exit', 'quit']:
                break
            if u:
                sys.stdout.write(f"\nAI: {ai.chat(u)}\n")
        except KeyboardInterrupt:
            break
