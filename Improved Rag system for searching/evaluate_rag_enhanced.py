"""
Enhanced RAG Evaluation Suite
================================
Metrics (all original + new):
  1.  Exact Match (EM)
  2.  Token F1 / Precision / Recall
  3.  Recall@K
  4.  MRR (Mean Reciprocal Rank)
  5.  BLEU-1
  6.  Context Precision
  7.  CRAG Quality Score   ← NEW: avg per-chunk CRAG confidence
  8.  CRAG Correct Rate    ← NEW: % chunks rated CORRECT by CRAG evaluator
  9.  Latency (ms)
  10. (Optional) LLM F1   ← NEW: F1 on the LLM-generated sub-answer

Run:
    python evaluate_rag_enhanced.py
    python evaluate_rag_enhanced.py --no-llm       # skip LLM (faster)
    python evaluate_rag_enhanced.py --no-bert      # use hash embedder
    python evaluate_rag_enhanced.py --sample 100   # larger sample
"""

import time, re, math, sys, argparse
from collections import Counter
from datasets import load_dataset
from rag_engine_enhanced import RAG, RAGConfig, RetrievalQuality

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sample",  type=int, default=50)
parser.add_argument("--k",       type=int, default=5)
parser.add_argument("--no-llm",  action="store_true")
parser.add_argument("--no-bert", action="store_true")
parser.add_argument("--no-crag", action="store_true")
args = parser.parse_args()

SAMPLE_SIZE = args.sample
RECALL_K    = args.k
RELEVANCE_THRESHOLD = 0.15

# ── Load dataset ───────────────────────────────────────────────────────────────
print("\n📦  Loading rag-mini-wikipedia …")
ds          = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
passages_ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
qa_pairs    = list(ds["test"])[:SAMPLE_SIZE]
passages    = passages_ds["passages"]

# ── Init enhanced RAG ──────────────────────────────────────────────────────────
cfg = RAGConfig(
    chunk_size    = 200,
    chunk_overlap = 30,
    use_bert      = not args.no_bert,
    use_llm       = not args.no_llm,
    use_crag      = not args.no_crag,
    use_mafrew    = True,
    final_k       = RECALL_K,
    bm25_w        = 0.3,
    vec_w         = 0.7,
    index_dir     = ".rag_enhanced_wiki_cache",
)
rag = RAG(cfg)

# ── Index if needed ────────────────────────────────────────────────────────────
stats = rag.stats()
if stats["chunks"] == 0:
    print(f"🔨  Indexing {len(passages)} passages …")
    t0 = time.perf_counter()
    for row in passages:
        rag.add_text(row["passage"], source=f"wiki:{row['id']}")
    print(f"✅  Indexed {rag.stats()['chunks']} chunks in {time.perf_counter()-t0:.1f}s")
    print(f"   BERT mode : {rag.stats()['bert_mode']}")
    print(f"   LLM mode  : {rag.stats()['llm_mode']}\n")
else:
    s = rag.stats()
    print(f"✅  Cached index — {s['chunks']} chunks  bert={s['bert_mode']}  llm={s['llm_mode']}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Metric helpers  (all original + new CRAG metrics)
# ══════════════════════════════════════════════════════════════════════════════

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def get_tokens(text): return normalize(text).split()

def exact_match(pred, ref):
    return 1.0 if normalize(ref) in normalize(pred) else 0.0

def token_f1(pred, ref):
    p_toks, r_toks = get_tokens(pred), get_tokens(ref)
    if not p_toks or not r_toks: return 0.0, 0.0, 0.0
    pc, rc = Counter(p_toks), Counter(r_toks)
    common = sum((pc & rc).values())
    if common == 0: return 0.0, 0.0, 0.0
    prec = common / len(p_toks)
    rec  = common / len(r_toks)
    f1   = 2 * prec * rec / (prec + rec)
    return prec, rec, f1

def bleu1(pred, ref):
    p_toks = get_tokens(pred)
    r_set  = set(get_tokens(ref))
    if not p_toks: return 0.0
    hits = sum(1 for t in p_toks if t in r_set)
    bp   = min(1.0, len(p_toks) / max(len(r_set), 1))
    return bp * (hits / len(p_toks))

def recall_at_k(hits, ref):
    ref_words = set(get_tokens(ref))
    if not ref_words: return 0.0
    for h in hits:
        ov = len(ref_words & set(get_tokens(h.chunk.text))) / len(ref_words)
        if ov >= RELEVANCE_THRESHOLD: return 1.0
    return 0.0

def mrr(hits, ref):
    ref_words = set(get_tokens(ref))
    if not ref_words: return 0.0
    for rank, h in enumerate(hits, 1):
        ov = len(ref_words & set(get_tokens(h.chunk.text))) / len(ref_words)
        if ov >= RELEVANCE_THRESHOLD: return 1.0 / rank
    return 0.0

def context_precision(hits, ref):
    if not hits: return 0.0
    ref_words = set(get_tokens(ref))
    if not ref_words: return 0.0
    rel = sum(1 for h in hits
              if len(ref_words & set(get_tokens(h.chunk.text))) / len(ref_words) >= RELEVANCE_THRESHOLD)
    return rel / len(hits)

# ── NEW: CRAG-specific metrics ─────────────────────────────────────────────────

def crag_avg_score(hits):
    """Average CRAG confidence score across retrieved chunks."""
    if not hits: return 0.0
    return sum(h.crag_score for h in hits) / len(hits)

def crag_correct_rate(hits):
    """Fraction of retrieved chunks rated CORRECT by the CRAG evaluator."""
    if not hits: return 0.0
    n_correct = sum(1 for h in hits if h.crag_quality == RetrievalQuality.CORRECT)
    return n_correct / len(hits)


# ══════════════════════════════════════════════════════════════════════════════
#  Run evaluation
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 76)
print(f"  ENHANCED RAG EVALUATION  —  {SAMPLE_SIZE} questions  —  Recall@{RECALL_K}")
print("=" * 76)

results = []

for i, row in enumerate(qa_pairs, 1):
    question  = row["question"]
    reference = row["answer"].strip()

    t0     = time.perf_counter()
    result = rag.query(question)
    latency_ms = (time.perf_counter() - t0) * 1000

    hits   = result["hits"]
    answer = result["answer"]

    # Split out LLM answer vs extractive answer from the formatted output
    llm_ans  = ""
    extr_ans = ""
    if "── LLM Answer" in answer:
        parts = answer.split("── LLM Answer")
        if len(parts) > 1:
            sub = parts[1].split("──")[1] if "──" in parts[1][10:] else parts[1]
            llm_ans = sub.strip()
    if "── Extractive Answer" in answer:
        parts = answer.split("── Extractive Answer")
        if len(parts) > 1:
            sub = parts[1].split("──")[1] if "──" in parts[1][10:] else parts[1]
            extr_ans = sub.strip()
    primary_answer = llm_ans if llm_ans else extr_ans if extr_ans else answer

    em          = exact_match(primary_answer, reference)
    p, r, f1    = token_f1(primary_answer, reference)
    b1          = bleu1(primary_answer, reference)
    rec_k       = recall_at_k(hits, reference)
    rr          = mrr(hits, reference)
    cp          = context_precision(hits, reference)
    crag_score  = crag_avg_score(hits)
    crag_rate   = crag_correct_rate(hits)

    # LLM F1 (only if we have a separate LLM answer)
    llm_f1 = token_f1(llm_ans, reference)[2] if llm_ans else None

    results.append({
        "question": question, "reference": reference,
        "em": em, "precision": p, "recall": r, "f1": f1,
        "bleu1": b1, "recall_k": rec_k, "mrr": rr,
        "ctx_prec": cp, "crag_score": crag_score, "crag_rate": crag_rate,
        "llm_f1": llm_f1, "latency_ms": latency_ms,
    })

    status = "✅" if f1 >= 0.3 else "❌"
    print(f"\n[{i:2d}/{SAMPLE_SIZE}] {status}  {question}")
    print(f"  Reference  : {reference[:100]}")
    print(f"  EM={em:.0f}  F1={f1:.2f}  Recall@{RECALL_K}={rec_k:.0f}  "
          f"MRR={rr:.2f}  CRAG_score={crag_score:.2f}  CRAG_correct={crag_rate:.2f}  "
          f"Lat={latency_ms:.0f}ms")
    if llm_f1 is not None:
        print(f"  LLM F1     : {llm_f1:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregate scores
# ══════════════════════════════════════════════════════════════════════════════

def avg(key):
    vals = [r[key] for r in results if r[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0

em_score    = avg("em")
f1_score    = avg("f1")
prec_score  = avg("precision")
rec_score   = avg("recall")
bleu_score  = avg("bleu1")
recall_k_sc = avg("recall_k")
mrr_score   = avg("mrr")
ctx_prec_sc = avg("ctx_prec")
crag_sc     = avg("crag_score")
crag_rate_sc= avg("crag_rate")
lat_score   = avg("latency_ms")

llm_f1_vals = [r["llm_f1"] for r in results if r["llm_f1"] is not None]
llm_f1_avg  = sum(llm_f1_vals) / len(llm_f1_vals) if llm_f1_vals else None

BAR_W = 30
def bar(v): return "█" * int(v * BAR_W) + "░" * (BAR_W - int(v * BAR_W))

print("\n" + "=" * 76)
print("  FINAL EVALUATION RESULTS")
print("=" * 76)
print(f"""
  Metric                     Score     Bar
  ──────────────────────────────────────────────────────────────────────
  Exact Match (EM)           {em_score:.3f}     {bar(em_score)}
  Token F1                   {f1_score:.3f}     {bar(f1_score)}
  Token Precision            {prec_score:.3f}     {bar(prec_score)}
  Token Recall               {rec_score:.3f}     {bar(rec_score)}
  BLEU-1                     {bleu_score:.3f}     {bar(bleu_score)}
  Recall@{RECALL_K}                  {recall_k_sc:.3f}     {bar(recall_k_sc)}
  MRR                        {mrr_score:.3f}     {bar(mrr_score)}
  Context Precision          {ctx_prec_sc:.3f}     {bar(ctx_prec_sc)}
  CRAG Avg Confidence ★      {crag_sc:.3f}     {bar(crag_sc)}
  CRAG Correct Rate  ★       {crag_rate_sc:.3f}     {bar(crag_rate_sc)}""")

if llm_f1_avg is not None:
    print(f"  LLM Answer F1      ★       {llm_f1_avg:.3f}     {bar(llm_f1_avg)}")

print(f"""  ──────────────────────────────────────────────────────────────────────
  Avg Latency                {lat_score:.1f} ms
  ★ = new enhanced metrics
""")

overall = (f1_score + recall_k_sc + mrr_score) / 3
if overall >= 0.7: grade = "A  🟢 Excellent"
elif overall >= 0.5: grade = "B  🟡 Good"
elif overall >= 0.35: grade = "C  🟠 Fair"
else: grade = "D  🔴 Needs improvement"

print(f"  Overall Grade  :  {grade}")
print(f"  Overall Score  :  {overall:.3f}  (avg of F1, Recall@K, MRR)")
print("=" * 76)

print("""
  METRIC EXPLANATIONS
  ──────────────────────────────────────────────────────────────────────
  Exact Match      Reference answer appears verbatim in output
  Token F1         Word-level overlap with reference answer
  Recall@K         Any retrieved chunk contains the answer keywords
  MRR              Mean Reciprocal Rank of first relevant chunk
  Context Prec     Fraction of retrieved chunks that are relevant
  CRAG Avg Score ★ Multi-signal (BM25+cosine) confidence per chunk
  CRAG Correct   ★ % of chunks CRAG rated as high-confidence CORRECT
  LLM F1         ★ Token F1 on the LLM sub-answer (T5/GPT-2) only
""")
