"""
SuperiorRAG Enhanced — Wikipedia Quickstart
============================================
Demonstrates the enhanced engine with:
  · BERT dense embeddings (sentence-transformers / raw BERT)
  · FLAN-T5 / GPT-2 answer generation (no API key)
  · CRAG corrective retrieval quality gating
  · MaFeRw multi-aspect query rewriting

Install:
    pip install numpy datasets transformers torch sentence-transformers

Run:
    python quickstart_wikipedia_enhanced.py
    python quickstart_wikipedia_enhanced.py --no-llm    # skip LLM (faster)
    python quickstart_wikipedia_enhanced.py --no-bert   # use hash embedder
"""

import time, sys, argparse
from datasets import load_dataset
from rag_engine_enhanced import RAG, RAGConfig, RetrievalQuality

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-llm",   action="store_true", help="Disable LLM generation")
parser.add_argument("--no-bert",  action="store_true", help="Use hash embedder instead of BERT")
parser.add_argument("--no-crag",  action="store_true", help="Disable CRAG evaluator")
parser.add_argument("--sample",   type=int, default=10, help="Number of QA pairs to test")
args = parser.parse_args()

# ── Load dataset ───────────────────────────────────────────────────────────────
print("\n📦  Loading rag-mini-wikipedia …")
ds          = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
passages_ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
qa_pairs    = ds["test"]
passages    = passages_ds["passages"]
print(f"   ✓ {len(passages)} passages  |  {len(qa_pairs)} Q&A pairs\n")

# ── Init enhanced RAG ──────────────────────────────────────────────────────────
print("⚙️   Initializing enhanced RAG engine …")
print(f"      BERT     : {'disabled' if args.no_bert else 'enabled (sentence-transformers → BERT → hash)'}")
print(f"      LLM      : {'disabled' if args.no_llm else 'enabled (FLAN-T5 → GPT-2 → KN-LM)'}")
print(f"      CRAG     : {'disabled' if args.no_crag else 'enabled (corrective retrieval)'}")
print(f"      MaFeRw   : enabled (multi-aspect query rewriting)\n")

cfg = RAGConfig(
    chunk_size    = 200,
    chunk_overlap = 30,
    use_bert      = not args.no_bert,
    use_llm       = not args.no_llm,
    use_crag      = not args.no_crag,
    use_mafrew    = True,
    final_k       = 5,
    bm25_w        = 0.35,
    vec_w         = 0.65,
    index_dir     = ".rag_enhanced_wiki_qs",
)
rag = RAG(cfg)

# ── Ingest ─────────────────────────────────────────────────────────────────────
stats = rag.stats()
if stats["chunks"] == 0:
    print(f"🔨  Indexing {len(passages)} passages …  (first run ~1-3 min with BERT)")
    t0 = time.perf_counter()
    for row in passages:
        rag.add_text(row["passage"], source=f"wiki:{row['id']}")
    elapsed = time.perf_counter() - t0
    s = rag.stats()
    print(f"\n✅  Indexed {s['chunks']} chunks in {elapsed:.1f}s")
    print(f"   BERT mode : {s['bert_mode']}")
    print(f"   LLM mode  : {s['llm_mode']}")
    print(f"   Vector RAM: {s['vector_mb']} MB\n")
else:
    s = rag.stats()
    print(f"✅  Cached index — {s['chunks']} chunks  bert={s['bert_mode']}  llm={s['llm_mode']}\n")


# ── Helper ─────────────────────────────────────────────────────────────────────
def ask(question: str, show_crag: bool = True) -> None:
    result = rag.query(question)
    sep = "─" * 74
    print(f"\n{sep}")
    print(f"❓  {question}")
    print(sep)
    # Print answer (strip long source block for readability)
    answer_lines = result["answer"].splitlines()
    printing = True
    for line in answer_lines:
        if "── Sources" in line:
            print(f"  {line}")
            printing = False
        if printing:
            print(f"  {line}")
        elif line.strip().startswith("["):
            # Print first 2 sources only
            print(f"  {line}")

    print(f"\n⏱  Latency: {result['latency_ms']} ms  |  {len(result['hits'])} chunks")

    if show_crag and result['hits']:
        quality_icons = {
            RetrievalQuality.CORRECT: "✅",
            RetrievalQuality.AMBIGUOUS: "🟡",
            RetrievalQuality.INCORRECT: "❌",
        }
        crag_line = "  CRAG: " + "  ".join(
            f"{quality_icons[h.crag_quality]} {h.crag_score:.2f}"
            for h in result['hits']
        )
        print(crag_line)
    print(sep)


# ── Sample Q&A evaluation ───────────────────────────────────────────────────────
SAMPLE_SIZE = args.sample
sample = list(qa_pairs)[:SAMPLE_SIZE]

print("=" * 74)
print(f"  SAMPLE QUERIES  —  {SAMPLE_SIZE} questions from rag-mini-wikipedia")
print("=" * 74)

correct_hits = 0
for i, row in enumerate(sample, 1):
    question  = row["question"]
    reference = row["answer"].strip().lower()

    result   = rag.query(question)
    answer   = result["answer"].lower()

    ref_words = set(reference.split())
    ans_words = set(answer.split())
    overlap   = len(ref_words & ans_words) / max(len(ref_words), 1)
    hit       = overlap >= 0.25
    correct_hits += hit

    # CRAG stats
    hits = result["hits"]
    crag_correct  = sum(1 for h in hits if h.crag_quality == RetrievalQuality.CORRECT)
    crag_avg      = sum(h.crag_score for h in hits) / len(hits) if hits else 0.0

    print(f"\n[{i:2d}/{SAMPLE_SIZE}] {'✅' if hit else '❌'}  {question}")
    print(f"  Reference  : {row['answer'][:120]}")
    if hits:
        preview = hits[0].chunk.text[:150].replace("\n", " ")
        print(f"  Top chunk  : {preview}…")
    print(f"  Latency    : {result['latency_ms']} ms  |  "
          f"CRAG correct: {crag_correct}/{len(hits)}  avg_score: {crag_avg:.2f}")

print(f"\n{'='*74}")
print(f"  Rough word-overlap accuracy on {SAMPLE_SIZE} questions: "
      f"{correct_hits}/{SAMPLE_SIZE}  ({100*correct_hits/SAMPLE_SIZE:.0f}%)")
print(f"{'='*74}\n")


# ── Interactive chat ───────────────────────────────────────────────────────────
print("💬  Enter your own questions (type 'quit' to exit):\n")
while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break
    if not q or q.lower() in {"quit", "exit", "q"}:
        break
    ask(q)
