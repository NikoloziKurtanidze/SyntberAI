import os
import sqlite3
import sys

class SyntberIngestor:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_dir, "syntber_knowledge.db")
        self.source_dir = os.path.join(self.base_dir, "import_here")
        
        if not os.path.exists(self.source_dir):
            os.makedirs(self.source_dir)
            
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS facts (answer TEXT NOT NULL)")
        conn.commit()
        conn.close()

    def run(self):
        sys.stdout.write("\n--- SYNTBER UNIVERSAL INGESTOR ---\n")
        files = [f for f in os.listdir(self.source_dir) if f.endswith(('.txt', '.md'))]
        
        if not files:
            sys.stdout.write(f"[*] Folder '{self.source_dir}' is empty.\n")
            sys.stdout.write("[*] Drop your .txt or .md files there and run this again.\n")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        count = 0
        for filename in files:
            sys.stdout.write(f"[*] Processing: {filename}\n")
            path = os.path.join(self.source_dir, filename)
            
            try:
                with open(path, 'r', encoding='utf-8') as f:

                    paragraphs = [p.strip() for p in f.read().split('\n\n') if len(p.strip()) > 15]
                    
                    for p in paragraphs:

                        clean_p = " ".join(p.split())
                        cursor.execute("INSERT INTO facts (answer) VALUES (?)", (clean_p,))
                        count += 1
            except Exception as e:
                sys.stdout.write(f"[!] Failed to read {filename}: {e}\n")
        
        conn.commit()
        conn.close()
        sys.stdout.write(f"\n[SUCCESS] {count} new knowledge segments added.\n")
        sys.stdout.write("[!] IMPORTANT: Run your indexer script now to refresh the vector brain.\n")

if __name__ == "__main__":
    SyntberIngestor().run()
