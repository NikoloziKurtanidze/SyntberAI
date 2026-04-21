Syntber AI v1.2.2Syntber is a locally-hosted RAG (Retrieval-Augmented Generation) AI engine designed for total privacy, speed, and mathematical precision. It combines Microsoft's Phi-3-mini weights with a custom architecture developed by The Syntber Project.
The engine features ExpDecayPE (Exponential Decay Positional Encoding) and is powered by a massive local knowledge base consisting of college-level textbooks and curated Wikipedia datasets.
 Key FeaturesTotal Privacy: Once set up, Syntber runs 100% offline.
GPU Massive Knowledge Base: Indexed with academic textbooks and Wikipedia data for broad general knowledge.
Acceleration: Support for NVIDIA GPUs (via CUDA) for lightning-fast response times.
Strict Logic Guardrails: Prioritizes user-provided variables (e.g., g=10) over pre-trained data.
Medical & Sensitive Data Filtering: Integrated safety patterns to block medical evaluations and filter sensitive content.
Plain-Text Math: Clean, human-readable math output (no LaTeX code). 
Installation & DependenciesInternet Requirement: You will need an active internet connection only for the initial setup (installing libraries and downloading models). Once the assets are on your machine, Syntber is fully offline.
1. Basic Installation (CPU)Bashpip install llama-cpp-python sentence-transformers faiss-cpu numpy tqdm
2. GPU Acceleration (NVIDIA/CUDA)If you have an NVIDIA GPU, install the version of llama-cpp-python with CUDA support for much faster performance:Bash# Example for Windows/Linux with CUDA 12.x
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
📂 Handling Large Files (GitHub 100MB Limit)Note: The .gguf model file and the .index file are significantly larger than 100MB and cannot be uploaded directly to a standard GitHub repository.To set up your local folder:Git LFS: You can use Git Large File Storage to track these files.Manual Download: Alternatively, download the Phi-3-mini-4k-instruct-Q4_K_M.gguf and your specific syntber_fast.index from your external storage link (e.g., Google Drive/Mega) and place them in the root folder.Required Folder Structure:Plaintext/SyntberAI
│   main.py                           <- Main engine
│   syntber_knowledge.db              <- SQLite DB (Books + Wikipedia)
│   syntber_fast.index                <- FAISS index (Large File)
│   Phi-3-mini-4k-instruct-Q4_K_M.gguf <- LLM Weights (Large File)
└───models/all-MiniLM-L6-v2/          <- Embedding Model
⚠️ Disclaimer & SafetySYNTBER IS AN AI AND CAN BE WRONG.Sensitive Data: The knowledge base contains scraped data from Wikipedia. While we have implemented filters to remove harmful content, users should exercise caution.Verify Information: Always verify mathematical, scientific, and factual outputs.Not a Professional: This software is not a substitute for professional medical, legal, or engineering advice.
⚖️ License & AttributionModel Weights: Microsoft (Phi-3-mini).
Architecture & Logic: The Syntber Project(Nikolozi)
Sources: Open-source textbooks and Wikipedia.Engine:
Powered by llama.cpp and SentenceTransformers.
