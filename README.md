# **DeepSeekRAG – Improved RAG System**

A simple search-and-answer tool built in Python. You give it some text files, ask a question, and it finds the most relevant parts and writes an answer. It runs on your own computer and needs **no API key**.

This is a RAG (Retrieval-Augmented Generation) project.

## What it does

1. Reads your documents (`.txt`, `.md`, `.html`, `.csv`, `.json`)
2. Splits them into small chunks
3. Finds the chunks that best match your question (keyword search + meaning-based search)
4. Checks if the chunks are actually good (CRAG idea)
5. Writes an answer using a small AI model (FLAN-T5 or GPT-2)

If the AI models are not installed, it still works using simpler built-in methods.

## Install

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Want the lightest setup? Just run `pip install numpy`.

## How to use

```bash
python rag_engine_enhanced.py                        # run the demo
python rag_engine_enhanced.py ingest notes.txt       # add your files
python rag_engine_enhanced.py query "What is RAG?"   # ask a question
python rag_engine_enhanced.py chat                   # chat mode
python rag_engine_enhanced.py clear                  # reset the index
```

Add `--no-llm` to a query for faster results (no AI model).

### Use it in your own code

```python
from rag_engine_enhanced import RAG, RAGConfig

rag = RAG(RAGConfig())
rag.add_text("Python is a programming language.", source="notes")

result = rag.query("What is Python?")
print(result["answer"])
```

## Test how good it is

```bash
pip install datasets
python evaluate_rag_enhanced.py --sample 50
```

This tests the system on Wikipedia questions and prints scores like F1, Recall@K and MRR.

## Files

| File | What it does |
|------|--------------|
| `rag_engine_enhanced.py` | Main code |
| `evaluate_rag_enhanced.py` | Measures accuracy |
| `quickstart_wikipedia_enhanced.py` | Wikipedia demo |
| `rag_web.py` | Version that searches the web live (optional) |

## Ideas used

- **CRAG** (Yan et al., 2024): checks whether retrieved text is good before using it.
- **MaFeRw** (Ma et al., 2025): improves the question so search works better.

These are simplified versions inspired by the papers.

## Note

The first run downloads small AI models (a few hundred MB). After that they are saved on your computer.

## License

MIT License. See the [LICENSE](LICENSE) file.
