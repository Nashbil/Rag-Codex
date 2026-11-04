# SuperiorRAG — Enhanced Edition
## BERT · FLAN-T5 · CRAG · MaFeRw · No API Key

---

## What's New vs the Original

| Component | Original | Enhanced |
|-----------|----------|----------|
| **Embedder** | TF-IDF × hashing trick | **sentence-transformers (MiniLM) → BERT → hash fallback** |
| **Generator** | KN n-gram LM + TextRank | **FLAN-T5 → GPT-2 → KN-LM** (local, no API) |
| **Retrieval quality** | Blind top-k | **CRAG corrective gating** (Yan et al. 2024) |
| **Query rewriting** | Static synonym expansion | **MaFeRw 3-signal expansion** (Ma et al. AAAI 2025) |
| **Reranker** | MMR | MMR (preserved) |

All original from-scratch components (BM25+, TextRank, Porter stemmer, KN-LM) are fully preserved as fallbacks.

---

## Install

```bash
# Minimal — still works with pure NumPy (uses hash embeddings + KN-LM)
pip install numpy

# Recommended — enables BERT + FLAN-T5
pip install numpy transformers torch sentence-transformers

# Full evaluation
pip install numpy transformers torch sentence-transformers datasets
```

---

## Quick Start

```bash
# Run the built-in demo (5 documents, 5 questions)
python rag_engine_enhanced.py

# Interactive chat
python rag_engine_enhanced.py chat

# Ingest your own files
python rag_engine_enhanced.py ingest notes.txt report.md

# Query
python rag_engine_enhanced.py query "What is the main finding?"

# Index info
python rag_engine_enhanced.py stats

# Skip LLM (faster, CPU-only, extractive only)
python rag_engine_enhanced.py query "..." --no-llm

# Skip CRAG evaluator
python rag_engine_enhanced.py query "..." --no-crag
```

---

## Wikipedia Evaluation

```bash
# Full enhanced evaluation (BERT + FLAN-T5 + CRAG + MaFeRw)
python quickstart_wikipedia_enhanced.py

# Faster (no LLM, BERT only)
python quickstart_wikipedia_enhanced.py --no-llm

# Fastest (no LLM, no BERT — pure classical)
python quickstart_wikipedia_enhanced.py --no-llm --no-bert

# Full metric evaluation suite
python evaluate_rag_enhanced.py
python evaluate_rag_enhanced.py --no-llm --sample 100
```

---

## Library Usage

```python
from rag_engine_enhanced import RAG, RAGConfig

cfg = RAGConfig(
    chunk_size    = 350,
    use_bert      = True,    # BERT dense embeddings
    use_llm       = True,    # FLAN-T5 / GPT-2 generation
    use_crag      = True,    # Corrective retrieval gating
    use_mafrew    = True,    # Multi-aspect query rewriting
    final_k       = 5,
    prefer_t5     = True,    # Prefer FLAN-T5 over GPT-2
)
rag = RAG(cfg)

rag.add_file("my_doc.txt")
rag.add_text("Any text here.", source="notes")

result = rag.query("What does the document say about X?")
print(result["answer"])          # LLM + extractive answer + CRAG stats
print(result["latency_ms"])

# Per-chunk CRAG quality
from rag_engine_enhanced import RetrievalQuality
for hit in result["hits"]:
    print(hit.chunk.text[:80], "→", hit.crag_quality.value, f"({hit.crag_score:.2f})")
```

---

## Paper Implementations

### 1. CRAG — Corrective Retrieval Augmented Generation
**Paper:** Yan et al., 2024. *Corrective Retrieval Augmented Generation.*
Findings of ICLR 2024 / cited in ACL 2024 proceedings.
**arXiv:** https://arxiv.org/abs/2401.15884

**Problem addressed:**
Naive RAG retrieves the top-k chunks unconditionally. If the retrieved
documents are irrelevant or partially wrong, the LLM produces answers
grounded in noise — hallucinations and factual errors.

**Original paper limitation:**
Requires a T5-based retrieval evaluator (fine-tuned on labeled data)
and external web search as the "corrective" step — not offline-friendly.

**Our improvements:**
- Evaluator uses BM25-overlap + cosine similarity — no fine-tuning, CPU-only.
- Three-mode gating (CORRECT ≥ 0.55 / AMBIGUOUS 0.25–0.55 / INCORRECT < 0.25).
- AMBIGUOUS chunks are refined: split into sentences, keep the top-50% by query overlap.
- INCORRECT chunks trigger a deeper BM25 re-retrieval pass (fully offline, no web search).
- New evaluation metrics: `CRAG Avg Confidence`, `CRAG Correct Rate`.

---

### 2. MaFeRw — Multi-Aspect Feedback Query Rewriting for RAG
**Paper:** Ma et al., AAAI 2025. *MaFeRw: Query Rewriting with Multi-Aspect
Feedbacks for Retrieval-Augmented Large Language Models.*
**arXiv:** https://arxiv.org/abs/2408.17072

**Problem addressed:**
T5-based query rewriters trained on context alone produce minimal improvement
because the training signal is sparse and the feedback loop is too loose.
A single context signal misses user intent embedded in retrieval and answer signals.

**Original paper limitation:**
Requires an expensive RL training loop with three separate reward models
and access to gold documents and ground-truth answers at training time.

**Our improvements:**
- Approximates the three MaFeRw feedback signals *at inference time* — no RL training needed.
- Aspect 1 (BM25 signal): highest-IDF terms from the top BM25-retrieved chunk.
- Aspect 2 (vector signal): highest-IDF terms from the top cosine-retrieved chunk.
- Aspect 3 (answer signal): TextRank summary terms from the top-3 chunks.
- All three aspects are fused, deduplicated, and used for a second retrieval pass.
- Results from both passes are merged with a 1.2× boost for overlapping IDs.
- Works fully offline, zero labels, zero RL overhead.

---

## Full Architecture (Enhanced)

```
Query
  │
  ▼
┌─────────────────────────────────────────────┐
│  MultiAspectQueryRewriter  (MaFeRw §14)     │
│  Aspect 1: BM25 high-IDF term injection     │
│  Aspect 2: vector nearest chunk keywords    │
│  Aspect 3: TextRank summary term extraction │
└─────────────────┬───────────────────────────┘
                  │  expanded query
                  ▼
┌─────────────────────────────────────────────┐
│  Hybrid Retrieval  (§§7-8, enhanced §6)     │
│  BM25+ sparse  +  BERT dense embeddings     │
│  Scores merged, dual-pass, boosted overlap  │
└─────────────────┬───────────────────────────┘
                  │  top-k candidates
                  ▼
┌─────────────────────────────────────────────┐
│  CRAGEvaluator  (§13)                       │
│  CORRECT   → pass through                  │
│  AMBIGUOUS → sentence-level refinement     │
│  INCORRECT → corrective BM25 re-retrieval  │
└─────────────────┬───────────────────────────┘
                  │  quality-gated hits
                  ▼
┌─────────────────────────────────────────────┐
│  MMR Reranker  (§15)                        │
│  Relevance × Diversity trade-off           │
└─────────────────┬───────────────────────────┘
                  │  final top-k hits
                  ▼
┌─────────────────────────────────────────────┐
│  AnswerGenerator  (§16)                     │
│  1. FLAN-T5  (encoder-decoder, local)       │
│  2. GPT-2    (decoder-only fallback)        │
│  3. KN-LM   (n-gram, zero-dep fallback)    │
│  + TextRank extractive answer (always)     │
└─────────────────────────────────────────────┘
                  │
                  ▼
           Answer + sources + CRAG stats
```

---

## Memory Footprint

| Corpus | Chunks | Hash RAM | BERT RAM | BERT+T5 RAM |
|--------|--------|----------|----------|-------------|
| 1 MB   | ~1,200 | ~2 MB    | ~4 MB    | ~650 MB     |
| 10 MB  | ~12,000| ~18 MB   | ~35 MB   | ~680 MB     |
| 100 MB | ~120K  | ~180 MB  | ~340 MB  | ~1.0 GB     |

**BERT models** (cached after first download, ~90 MB for MiniLM-L6):
- `all-MiniLM-L6-v2` — 22M params, 384-dim, fastest
- `bert-base-uncased` — 110M params, 768-dim, fallback

**Generator models** (cached after first download):
- `google/flan-t5-small` — 60M params, ~300 MB
- `gpt2` — 117M params, ~500 MB
- `KN-LM` — 0 MB (built from corpus at runtime)

---

## Compatibility

All original CLI commands work unchanged:
```bash
python rag_engine_enhanced.py ingest notes.txt
python rag_engine_enhanced.py query "What is X?"
python rag_engine_enhanced.py chat
python rag_engine_enhanced.py stats
python rag_engine_enhanced.py clear
```

The index format is **not** backwards-compatible with the original engine
(embed_dim changes when BERT is enabled). Use `clear` to reset if switching.
