"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   SuperiorRAG  ·  ENHANCED  EDITION                                         ║
║                                                                              ║
║   NEW vs original:                                                           ║
║   ✦ T5 / FLAN-T5-small  encoder-decoder  for answer generation              ║
║   ✦ BERT (sentence-transformers-style) dense embeddings                     ║
║   ✦ Open-source LLM  (GPT-2 / FLAN-T5)  — no API key, local inference      ║
║   ✦ CRAG-style Corrective Retrieval  (Yan et al., 2024 — ACL findings)     ║
║       → Addresses: static retrieval ignores quality of retrieved context    ║
║       → Fix: lightweight relevance classifier gates & corrects retrieval    ║
║   ✦ MaFeRw-inspired Multi-Aspect Query Rewriting  (AAAI 2025)              ║
║       → Addresses: single-context rewriter misses user intent signals       ║
║       → Fix: rewrite using BM25 signal + vector similarity + answer signal  ║
║   ✦ All original scratch components preserved as CPU-only fallback          ║
║                                                                              ║
║   Install deps (optional but recommended):                                   ║
║     pip install transformers torch sentence-transformers                     ║
║                                                                              ║
║   Run demo:                                                                  ║
║     python rag_engine_enhanced.py                                            ║
║     python rag_engine_enhanced.py query "What is RAG?"                      ║
║     python rag_engine_enhanced.py chat                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

PAPER REFERENCES IMPLEMENTED
──────────────────────────────────────────────────────────────────────────────
1. CRAG — "Corrective Retrieval Augmented Generation"
   Yan et al., 2024, Findings of ACL (ICLR 2024 proceedings crosslisted).
   https://openreview.net/abs/2401.15884

   Key limitation addressed:
     Naive RAG blindly trusts all retrieved chunks even when they are
     semantically irrelevant or contradictory to the query.  CRAG introduces
     a lightweight "retrieval evaluator" that assigns each retrieved chunk a
     confidence score; low-confidence chunks trigger corrective retrieval
     (web search or a broader index scan) or are filtered out entirely.

   Our implementation:
     - `CRAGEvaluator`: BM25-overlap + cosine similarity → per-chunk score
     - Three-mode gating: CORRECT (≥0.6) / AMBIGUOUS (0.3-0.6) / INCORRECT (<0.3)
     - INCORRECT chunks are replaced; AMBIGUOUS chunks get query-expansion boost
     - No external web search needed — uses deeper BM25 pass as "corrective" step

2. MaFeRw — "Query Rewriting with Multi-Aspect Feedbacks for RAG"
   Ma et al., AAAI 2025.  arXiv:2408.17072

   Key limitation addressed:
     T5-based query rewriters trained only on gold-document context
     produce minimal improvement because the rewriting signal is too
     sparse and far from the final answer quality.  A single feedback
     source (context) cannot capture multi-aspect user intent.

   Our implementation:
     `MultiAspectQueryRewriter`:
     - Aspect 1 (BM25 signal):  expand query with top BM25 hit terms
     - Aspect 2 (vector signal): append semantically nearest chunk keywords
     - Aspect 3 (answer signal): extract high-IDF terms from TextRank summary
     - Rewrites are fused and deduplicated before final retrieval
     - Works WITHOUT an LLM for rewriting (can optionally use T5)
"""

from __future__ import annotations

# ─── stdlib ──────────────────────────────────────────────────────────────────
import re, os, sys, csv, json, gzip, math, time, html, array, heapq, pickle
import hashlib, logging, textwrap, argparse, threading, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RAG+")

# ─── optional deep-learning imports ──────────────────────────────────────────
_HAS_TRANSFORMERS = False
_HAS_TORCH = False
_HAS_ST = False          # sentence-transformers

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        T5ForConditionalGeneration,
        T5Tokenizer,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        pipeline,
    )
    _HAS_TRANSFORMERS = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    pass

_DEVICE = "cpu"
if _HAS_TORCH and torch.cuda.is_available():
    _DEVICE = "cuda"
elif _HAS_TORCH:
    try:
        if torch.backends.mps.is_available():
            _DEVICE = "mps"
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  §1  PORTER STEMMER  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

class PorterStemmer:
    _VOWELS = frozenset("aeiou")

    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 2:
            return word
        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5a(word)
        word = self._step5b(word)
        return word

    def _cons(self, word, i):
        c = word[i]
        if c in self._VOWELS: return False
        if c == "y": return i == 0 or not self._cons(word, i - 1)
        return True

    def _m(self, word):
        n, i, m = len(word), 0, 0
        while i < n and self._cons(word, i): i += 1
        while i < n:
            while i < n and not self._cons(word, i): i += 1
            m += 1
            while i < n and self._cons(word, i): i += 1
        return m

    def _has_vowel(self, stem):
        return any(not self._cons(stem, i) for i in range(len(stem)))

    def _ends_double_cons(self, word):
        return (len(word) >= 2 and word[-1] == word[-2]
                and self._cons(word, len(word) - 1))

    def _cvc(self, word):
        n = len(word)
        if n < 3: return False
        return (self._cons(word, n - 3) and not self._cons(word, n - 2)
                and self._cons(word, n - 1) and word[-1] not in "wxy")

    def _step1a(self, w):
        if w.endswith("sses"): return w[:-2]
        if w.endswith("ies"):  return w[:-2]
        if w.endswith("ss"):   return w
        if w.endswith("s"):    return w[:-1]
        return w

    def _step1b(self, w):
        if w.endswith("eed"):
            if self._m(w[:-3]) > 0: return w[:-1]
            return w
        flag = False
        for suf in ("ed", "ing"):
            if w.endswith(suf) and self._has_vowel(w[:-len(suf)]):
                w = w[:-len(suf)]; flag = True; break
        if flag:
            for suf2, rep in [("at","ate"),("bl","ble"),("iz","ize")]:
                if w.endswith(suf2): return w + rep[2:]
            if self._ends_double_cons(w) and w[-1] not in "lsz": return w[:-1]
            if self._m(w) == 1 and self._cvc(w): return w + "e"
        return w

    def _step1c(self, w):
        if w.endswith("y") and self._has_vowel(w[:-1]): return w[:-1] + "i"
        return w

    def _step2(self, w):
        for suf, rep in [
            ("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
            ("izer","ize"),("abli","able"),("alli","al"),("entli","ent"),
            ("eli","e"),("ousli","ous"),("ization","ize"),("ation","ate"),
            ("ator","ate"),("alism","al"),("iveness","ive"),("fulness","ful"),
            ("ousness","ous"),("aliti","al"),("iviti","ive"),("biliti","ble"),
        ]:
            if w.endswith(suf) and self._m(w[:-len(suf)]) > 0:
                return w[:-len(suf)] + rep
        return w

    def _step3(self, w):
        for suf, rep in [
            ("icate","ic"),("ative",""),("alize","al"),("iciti","ic"),
            ("ical","ic"),("ful",""),("ness",""),
        ]:
            if w.endswith(suf) and self._m(w[:-len(suf)]) > 0:
                return w[:-len(suf)] + rep
        return w

    def _step4(self, w):
        for suf in ["al","ance","ence","er","ic","able","ible","ant",
                    "ement","ment","ent","ion","ou","ism","ate","iti","ous","ive","ize"]:
            if w.endswith(suf):
                stem = w[:-len(suf)]
                if suf == "ion":
                    if self._m(stem) > 1 and stem and stem[-1] in "st": return stem
                elif self._m(stem) > 1: return stem
        return w

    def _step5a(self, w):
        if w.endswith("e"):
            stem = w[:-1]; m = self._m(stem)
            if m > 1 or (m == 1 and not self._cvc(stem)): return stem
        return w

    def _step5b(self, w):
        if self._m(w) > 1 and self._ends_double_cons(w) and w.endswith("l"): return w[:-1]
        return w


# ══════════════════════════════════════════════════════════════════════════════
#  §2  TOKENIZER  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

_STOPS = frozenset("""
a about above after again against all also although am an and any are aren't as
at be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each even ever few
for from further get got had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's however i i'd i'll
i'm i've if in into is isn't it it's its itself just let's me more most mustn't
my myself new no nor not now of off on once only or other ought our ours
ourselves out over own re same shan't she she'd she'll she's should shouldn't
since so some such than that that's the their theirs them themselves then there
there's these they they'd they'll they're they've this those through to too
under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with
won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:[-'][a-zA-Z0-9]+)*\b")
_stemmer = PorterStemmer()


def tokenize(text: str, stem: bool = True, stops: bool = True) -> List[str]:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if stops:
        words = [w for w in words if w not in _STOPS and len(w) > 1]
    if stem:
        words = [_stemmer.stem(w) for w in words]
    return words


def tokenize_raw(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if len(w) > 0]


# ══════════════════════════════════════════════════════════════════════════════
#  §3  DOCUMENT LOADER  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Document:
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(
                f"{self.source}:{self.text[:128]}".encode()
            ).hexdigest()[:12]


class Loader:
    @classmethod
    def load(cls, path: str | Path) -> List[Document]:
        p = Path(path)
        methods = {
            ".txt": cls._txt, ".md": cls._txt, ".rst": cls._txt,
            ".html": cls._html, ".htm": cls._html,
            ".csv": cls._csv,
            ".json": cls._json, ".jsonl": cls._jsonl,
        }
        fn = methods.get(p.suffix.lower(), cls._txt)
        try:
            docs = fn(p)
            log.info(f"Loaded {len(docs)} doc(s) ← {p.name}")
            return docs
        except Exception as e:
            log.warning(f"Load error {p}: {e}")
            return []

    @classmethod
    def load_string(cls, text: str, source: str = "<string>") -> Document:
        return Document(text=text, source=source)

    @staticmethod
    def _txt(p): return [Document(text=p.read_text(errors="replace"), source=str(p))]

    @staticmethod
    def _html(p):
        raw = p.read_text(errors="replace")
        raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        return [Document(text=re.sub(r"\s+", " ", text).strip(), source=str(p))]

    @staticmethod
    def _csv(p):
        rows = []
        with open(p, newline="", errors="replace") as f:
            for row in csv.DictReader(f):
                rows.append("  ".join(f"{k}: {v}" for k, v in row.items() if v.strip()))
        return [Document(text="\n".join(rows), source=str(p))]

    @staticmethod
    def _json(p):
        return [Document(text=json.dumps(json.loads(p.read_text()), indent=2), source=str(p))]

    @staticmethod
    def _jsonl(p):
        docs = []
        for i, line in enumerate(p.read_text().splitlines()):
            if line.strip():
                docs.append(Document(text=json.dumps(json.loads(line)), source=str(p), metadata={"line": i}))
        return docs


# ══════════════════════════════════════════════════════════════════════════════
#  §4  SEMANTIC CHUNKER  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    text: str
    source: str
    doc_id: str
    chunk_id: str
    idx: int
    tokens: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Chunker:
    _SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[A-Z\"\'])|(?<=\n)")
    _PARA_SPLIT = re.compile(r"\n{2,}|(?=^#{1,3} )", re.M)

    def __init__(self, size: int = 350, overlap: int = 60, min_size: int = 40):
        self.size = size
        self.overlap = overlap
        self.min_size = min_size

    def chunk(self, doc: Document) -> List[Chunk]:
        paragraphs = [p.strip() for p in self._PARA_SPLIT.split(doc.text) if p.strip()]
        sentences: List[str] = []
        for para in paragraphs:
            parts = [s.strip() for s in self._SENT_END.split(para) if s.strip()]
            sentences.extend(parts if parts else [para])

        groups: List[str] = []
        buf: List[str] = []
        buf_wc = 0

        for sent in sentences:
            wc = len(sent.split())
            if buf_wc + wc > self.size and buf:
                groups.append(" ".join(buf))
                overlap_words: List[str] = []
                ow = 0
                for s in reversed(buf):
                    sw = len(s.split())
                    if ow + sw > self.overlap: break
                    overlap_words.insert(0, s)
                    ow += sw
                buf = overlap_words
                buf_wc = ow
            buf.append(sent)
            buf_wc += wc

        if buf:
            groups.append(" ".join(buf))

        chunks: List[Chunk] = []
        for i, text in enumerate(groups):
            if len(text.split()) < self.min_size:
                continue
            cid = hashlib.md5(f"{doc.doc_id}:{i}:{text[:64]}".encode()).hexdigest()[:16]
            chunks.append(Chunk(
                text=text, source=doc.source,
                doc_id=doc.doc_id, chunk_id=cid, idx=i,
                metadata=dict(doc.metadata),
            ))
        return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  §5  HASH EMBEDDER  (original fallback, preserved)
# ══════════════════════════════════════════════════════════════════════════════

class HashEmbedder:
    """TF-IDF × hashing trick → random projection → dense float32 vector."""

    def __init__(self, dim: int = 256, n_buckets: int = 2**17):
        self.dim = dim
        self.n_buckets = n_buckets
        self.idf: Dict[str, float] = {}
        rng = np.random.default_rng(42)
        self._proj = rng.standard_normal((n_buckets, dim)).astype(np.float32)
        norms = np.linalg.norm(self._proj, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        self._proj /= norms

    def _hash(self, tok: str) -> int:
        return int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.n_buckets

    def fit(self, token_lists: List[List[str]]) -> None:
        N = len(token_lists) or 1
        df: Counter = Counter()
        for tl in token_lists:
            df.update(set(tl))
        self.idf = {t: math.log(1 + N / (c + 1)) for t, c in df.items()}

    def embed(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        tf = Counter(tokens)
        n = len(tokens)
        vec = np.zeros(self.n_buckets, dtype=np.float32)
        for tok, cnt in tf.items():
            weight = (cnt / n) * self.idf.get(tok, 1.0)
            vec[self._hash(tok)] += weight
        for i in range(min(len(tokens) - 1, 512)):
            bg = f"{tokens[i]}\x00{tokens[i+1]}"
            weight = 0.4 * self.idf.get(tokens[i], 1.0) * self.idf.get(tokens[i+1], 1.0)
            vec[self._hash(bg)] += weight
        n2 = np.linalg.norm(vec)
        vec = vec / n2 if n2 > 1e-9 else vec
        out = vec @ self._proj
        n3 = np.linalg.norm(out)
        return out / n3 if n3 > 1e-9 else out


# ══════════════════════════════════════════════════════════════════════════════
#  §6  BERT DENSE EMBEDDER  (NEW — uses sentence-transformers or raw BERT)
# ══════════════════════════════════════════════════════════════════════════════

class BERTEmbedder:
    """
    Dense sentence embedder using:
      - sentence-transformers (all-MiniLM-L6-v2) if available  → 384-dim
      - raw HuggingFace BERT (mean-pooling) if only transformers available
      - Falls back to HashEmbedder if no deep-learning libs present

    The model is loaded lazily on first call and cached.
    """

    _MODEL_NAME_ST = "all-MiniLM-L6-v2"
    _MODEL_NAME_HF = "bert-base-uncased"

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._model = None
        self._tokenizer = None
        self._mode: str = "none"   # "st" | "hf" | "hash"
        self._hash_fallback: Optional[HashEmbedder] = None
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        self._loaded = True
        global _HAS_ST, _HAS_TRANSFORMERS, _HAS_TORCH

        if _HAS_ST:
            try:
                log.info("Loading sentence-transformers (all-MiniLM-L6-v2) …")
                self._model = SentenceTransformer(self._MODEL_NAME_ST, device=_DEVICE)
                self.dim = self._model.get_sentence_embedding_dimension()
                self._mode = "st"
                log.info(f"✅ sentence-transformers loaded  dim={self.dim}  device={_DEVICE}")
                return
            except Exception as e:
                log.warning(f"sentence-transformers load failed: {e}")

        if _HAS_TRANSFORMERS and _HAS_TORCH:
            try:
                log.info("Loading BERT (bert-base-uncased) for dense embeddings …")
                self._tokenizer = AutoTokenizer.from_pretrained(self._MODEL_NAME_HF)
                self._model = AutoModel.from_pretrained(self._MODEL_NAME_HF).to(_DEVICE)
                self._model.eval()
                self.dim = 768
                self._mode = "hf"
                log.info(f"✅ BERT loaded  dim={self.dim}  device={_DEVICE}")
                return
            except Exception as e:
                log.warning(f"BERT load failed: {e}")

        log.info("No transformer libs — using HashEmbedder fallback.")
        self._hash_fallback = HashEmbedder(self.dim)
        self._mode = "hash"

    def fit(self, token_lists: List[List[str]]) -> None:
        """Only needed for hash fallback; BERT is self-supervised."""
        self._load()
        if self._mode == "hash" and self._hash_fallback:
            self._hash_fallback.fit(token_lists)

    def embed(self, text_or_tokens) -> np.ndarray:
        self._load()
        if isinstance(text_or_tokens, list):
            text = " ".join(text_or_tokens)
        else:
            text = str(text_or_tokens)

        if self._mode == "st":
            vec = self._model.encode(text, convert_to_numpy=True,
                                     show_progress_bar=False, normalize_embeddings=True)
            return vec.astype(np.float32)

        elif self._mode == "hf":
            return self._bert_embed(text)

        else:
            # hash fallback — embed tokenized form
            toks = tokenize(text) if isinstance(text_or_tokens, str) else text_or_tokens
            if self._hash_fallback:
                return self._hash_fallback.embed(toks)
            return np.zeros(self.dim, dtype=np.float32)

    def _bert_embed(self, text: str) -> np.ndarray:
        with torch.no_grad():
            enc = self._tokenizer(
                text, return_tensors="pt",
                max_length=512, truncation=True, padding=True
            ).to(_DEVICE)
            out = self._model(**enc)
            # Mean pooling
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            vec = vec[0].cpu().numpy().astype(np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-9 else vec

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._load()
        if self._mode == "st":
            vecs = self._model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True,
                show_progress_bar=False, normalize_embeddings=True
            )
            return vecs.astype(np.float32)
        return np.stack([self.embed(t) for t in texts])


# ══════════════════════════════════════════════════════════════════════════════
#  §7  VECTOR INDEX  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._ids: List[str] = []
        self._f32: Optional[np.ndarray] = None
        self._q8: Optional[np.ndarray] = None
        self._scale = 1.0
        self._zero = 0.0
        self._stale = False

    def add(self, cid: str, vec: np.ndarray) -> None:
        v = vec.astype(np.float32).reshape(1, -1)
        self._f32 = v if self._f32 is None else np.vstack([self._f32, v])
        self._ids.append(cid)
        self._stale = True

    def commit(self) -> None:
        if self._f32 is None or not self._stale: return
        mn, mx = float(self._f32.min()), float(self._f32.max())
        rng = mx - mn if mx != mn else 1.0
        self._scale = rng / 255.0
        self._zero = mn
        self._q8 = ((self._f32 - mn) / rng * 255).clip(0, 255).astype(np.uint8)
        self._f32 = None
        self._stale = False

    def search(self, query: np.ndarray, k: int) -> List[Tuple[str, float]]:
        if self._q8 is None or not self._ids: return []
        if self._stale: self.commit()
        mat = self._q8.astype(np.float32) * self._scale + self._zero
        scores = mat @ query.astype(np.float32)
        k = min(k, len(self._ids))
        if k == 0: return []
        top_idx = (np.argsort(scores)[::-1] if k == len(self._ids)
                   else np.argpartition(scores, -k)[-k:])
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def memory_mb(self) -> float:
        if self._q8 is not None: return self._q8.nbytes / 1024 / 1024
        if self._f32 is not None: return self._f32.nbytes / 1024 / 1024
        return 0.0

    def __len__(self): return len(self._ids)


# ══════════════════════════════════════════════════════════════════════════════
#  §8  BM25+ INDEX  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75, delta: float = 1.0):
        self.k1 = k1; self.b = b; self.delta = delta
        self._ids: List[str] = []
        self._toks: List[List[str]] = []
        self._idf: Dict[str, float] = {}
        self._tf_docs: List[Counter] = []
        self._avgdl = 1.0
        self._built = False

    def add(self, cid: str, tokens: List[str]) -> None:
        self._ids.append(cid)
        self._toks.append(tokens)
        self._built = False

    def build(self) -> None:
        N = len(self._toks)
        if N == 0: return
        dl = [len(t) for t in self._toks]
        self._avgdl = sum(dl) / N or 1.0
        df: Counter = Counter()
        for t in self._toks: df.update(set(t))
        self._idf = {
            term: math.log(1 + (N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._tf_docs = [Counter(t) for t in self._toks]
        self._built = True

    def search(self, query_tokens: List[str], k: int) -> List[Tuple[str, float]]:
        if not self._built: self.build()
        if not self._ids: return []
        scores = np.zeros(len(self._ids), dtype=np.float64)
        for term in set(query_tokens):
            if term not in self._idf: continue
            idf = self._idf[term]
            for i, (tf_d, toks) in enumerate(zip(self._tf_docs, self._toks)):
                tf = tf_d.get(term, 0)
                if tf == 0: continue
                dl = len(toks)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
                scores[i] += idf * (self.delta + tf_norm)
        k = min(k, len(self._ids))
        if k == 0: return []
        top = (np.argsort(scores)[::-1] if k == len(self._ids)
               else np.argpartition(scores, -k)[-k:])
        top = top[np.argsort(scores[top])[::-1]]
        return [(self._ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def __len__(self): return len(self._ids)


# ══════════════════════════════════════════════════════════════════════════════
#  §9  TFIDF MATRIX  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

class TFIDFMatrix:
    def __init__(self):
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray = np.array([])
        self._mat: Optional[np.ndarray] = None

    def fit_transform(self, token_lists: List[List[str]]) -> np.ndarray:
        all_terms = sorted({t for tl in token_lists for t in tl})
        self._vocab = {t: i for i, t in enumerate(all_terms)}
        V, N = len(all_terms), len(token_lists)
        mat = np.zeros((N, V), dtype=np.float32)
        for i, toks in enumerate(token_lists):
            c = Counter(toks); total = len(toks) or 1
            for t, cnt in c.items():
                if t in self._vocab: mat[i, self._vocab[t]] = cnt / total
        df = (mat > 0).sum(axis=0) + 1
        self._idf = np.log((N + 1) / df).astype(np.float32)
        mat = mat * self._idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1.0
        self._mat = mat / norms
        return self._mat

    def transform(self, tokens: List[str]) -> np.ndarray:
        V = len(self._vocab)
        vec = np.zeros(V, dtype=np.float32)
        c = Counter(tokens); total = len(tokens) or 1
        for t, cnt in c.items():
            if t in self._vocab: vec[self._vocab[t]] = cnt / total
        vec *= self._idf
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-9 else vec


# ══════════════════════════════════════════════════════════════════════════════
#  §10  TEXTRANK EXTRACTIVE SUMMARISER  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

class TextRank:
    def __init__(self, damping: float = 0.85, iterations: int = 30):
        self.d = damping; self.iters = iterations

    def summarise(self, text: str, n_sentences: int = 3) -> str:
        sents = self._split_sentences(text)
        if len(sents) <= n_sentences: return text
        tfidf = TFIDFMatrix()
        mat = tfidf.fit_transform([tokenize(s) for s in sents])
        sim = mat @ mat.T
        np.fill_diagonal(sim, 0.0)
        row_sum = sim.sum(axis=1, keepdims=True)
        row_sum[row_sum < 1e-9] = 1.0
        trans = sim / row_sum
        scores = np.ones(len(sents), dtype=np.float32) / len(sents)
        for _ in range(self.iters):
            scores = (1 - self.d) / len(sents) + self.d * (trans.T @ scores)
        top_idx = sorted(np.argsort(scores)[-n_sentences:])
        return " ".join(sents[i] for i in top_idx)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if len(p.split()) > 3]


# ══════════════════════════════════════════════════════════════════════════════
#  §11  KNESER-NEY TRIGRAM LM  (original, preserved as fallback)
# ══════════════════════════════════════════════════════════════════════════════

class KNLanguageModel:
    DISCOUNT = 0.75

    def __init__(self):
        self._tri = Counter(); self._bi = Counter()
        self._uni = Counter(); self._vocab = set()
        self._trained = False

    def train(self, texts: List[str]) -> None:
        self._tri.clear(); self._bi.clear(); self._uni.clear()
        for text in texts:
            words = ["<s>", "<s>"] + tokenize_raw(text) + ["</s>"]
            for i in range(2, len(words)):
                w1, w2, w3 = words[i-2], words[i-1], words[i]
                self._tri[(w1, w2, w3)] += 1
                self._bi[(w1, w2)] += 1
                self._uni[w3] += 1
                self._vocab.add(w3)
        self._trained = True

    def generate(self, seed: str, max_words: int = 80, beam: int = 4) -> str:
        if not self._trained or not self._vocab:
            return "(language model not trained)"
        seed_words = tokenize_raw(seed)[-2:] if seed else []
        w1 = seed_words[-2] if len(seed_words) >= 2 else "<s>"
        w2 = seed_words[-1] if len(seed_words) >= 1 else "<s>"
        beams: List[Tuple[float, List[str]]] = [(0.0, [w1, w2])]
        completed = []
        for _ in range(max_words):
            candidates = []
            for log_p, seq in beams:
                w1_, w2_ = seq[-2], seq[-1]
                probs = self._next_probs(w1_, w2_)
                for w3, p in sorted(probs.items(), key=lambda x: -x[1])[:beam * 2]:
                    new_lp = log_p + math.log(p + 1e-30)
                    if w3 == "</s>": completed.append((new_lp, seq + [w3]))
                    else: candidates.append((new_lp, seq + [w3]))
            if not candidates: break
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[:beam]
        beams.extend(completed)
        if not beams: return "(no output)"
        best = max(beams, key=lambda x: x[0] / max(len(x[1]), 1))
        words = [w for w in best[1][2:] if w not in ("<s>", "</s>")]
        return self._detokenize(words)

    def _next_probs(self, w1, w2):
        D = self.DISCOUNT
        bi_cnt = self._bi.get((w1, w2), 0)
        probs = {}
        if bi_cnt > 0:
            for (a, b, w3), cnt in self._tri.items():
                if a == w1 and b == w2:
                    p_tri = max(cnt - D, 0) / bi_cnt
                    bi2 = self._bi.get((w2, w3), 0)
                    uni_cnt = self._uni.get(w3, 0)
                    p_bi = max(bi2 - D, 0) / max(self._bi.get((w2,), 0) or 1, 1)
                    p_uni = uni_cnt / max(sum(self._uni.values()), 1)
                    lam_bi = D * len({w for (a_,b_),_ in self._bi.items() if a_==w2}) / max(bi_cnt,1)
                    lam_uni = D * len({w for (a_,b_) in self._bi.keys() if a_==w1 and b_==w2}) / max(bi_cnt,1)
                    probs[w3] = p_tri + lam_uni * (p_bi + lam_bi * p_uni)
        else:
            total_uni = sum(self._uni.values()) or 1
            for w, cnt in self._uni.most_common(50): probs[w] = cnt / total_uni
        s = sum(probs.values()) or 1.0
        return {w: v / s for w, v in probs.items()}

    @staticmethod
    def _detokenize(words):
        if not words: return ""
        result = words[0].capitalize()
        for w in words[1:]:
            result += w if w in ".,;:!?" else " " + w
        result = re.sub(r"([.!?])\s+([a-z])", lambda m: m.group(1)+" "+m.group(2).upper(), result)
        if result and result[-1] not in ".!?": result += "."
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  §12  OPEN-SOURCE LLM GENERATOR  (NEW — FLAN-T5 or GPT-2, no API key)
# ══════════════════════════════════════════════════════════════════════════════

class OpenSourceLLM:
    """
    Local open-source LLM for answer generation.

    Priority:
      1. google/flan-t5-small   — encoder-decoder, instruction-tuned, ~300 MB
      2. gpt2                   — decoder-only, ~500 MB
      3. Falls back to KN n-gram LM (zero deps)

    Both models run fully offline after the first download.
    No API keys, no external calls at inference time.
    """

    _FLAN_T5_MODEL = "google/flan-t5-small"
    _GPT2_MODEL    = "gpt2"

    def __init__(self, prefer_t5: bool = True, max_new_tokens: int = 200):
        self._prefer_t5 = prefer_t5
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._mode: str = "none"   # "flan-t5" | "gpt2" | "kn"
        self._kn = KNLanguageModel()
        self._loaded = False

    def _load(self):
        if self._loaded: return
        self._loaded = True

        if not (_HAS_TRANSFORMERS and _HAS_TORCH):
            log.info("No transformers/torch — using KN-LM for generation.")
            self._mode = "kn"
            return

        # Try FLAN-T5 first
        if self._prefer_t5:
            try:
                log.info(f"Loading FLAN-T5 ({self._FLAN_T5_MODEL}) …")
                self._tokenizer = T5Tokenizer.from_pretrained(
                    self._FLAN_T5_MODEL, legacy=False)
                self._model = T5ForConditionalGeneration.from_pretrained(
                    self._FLAN_T5_MODEL).to(_DEVICE)
                self._model.eval()
                self._mode = "flan-t5"
                log.info(f"✅ FLAN-T5 loaded  device={_DEVICE}")
                return
            except Exception as e:
                log.warning(f"FLAN-T5 load failed ({e}) — trying GPT-2 …")

        # GPT-2 fallback
        try:
            log.info(f"Loading GPT-2 ({self._GPT2_MODEL}) …")
            self._tokenizer = GPT2Tokenizer.from_pretrained(self._GPT2_MODEL)
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = GPT2LMHeadModel.from_pretrained(self._GPT2_MODEL).to(_DEVICE)
            self._model.eval()
            self._mode = "gpt2"
            log.info(f"✅ GPT-2 loaded  device={_DEVICE}")
        except Exception as e:
            log.warning(f"GPT-2 load failed ({e}) — using KN-LM.")
            self._mode = "kn"

    def generate(self, query: str, context: str, kn_texts: List[str] = None) -> str:
        """Generate an answer given query + retrieved context."""
        self._load()

        if self._mode == "flan-t5":
            return self._generate_flan_t5(query, context)
        elif self._mode == "gpt2":
            return self._generate_gpt2(query, context)
        else:
            if kn_texts:
                self._kn.train(kn_texts)
            return self._kn.generate(query)

    def _generate_flan_t5(self, query: str, context: str) -> str:
        """
        FLAN-T5 is an encoder-decoder trained on a mixture of tasks including
        extractive QA. We feed it an instruction prompt with context.
        """
        # Truncate context to stay within T5's 512 token limit
        ctx_words = context.split()[:350]
        ctx = " ".join(ctx_words)

        prompt = (
            f"Answer the question based on the context below.\n\n"
            f"Context: {ctx}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        with torch.no_grad():
            inputs = self._tokenizer(
                prompt, return_tensors="pt",
                max_length=512, truncation=True
            ).to(_DEVICE)
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
            )
        answer = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip() if answer.strip() else "(no answer generated)"

    def _generate_gpt2(self, query: str, context: str) -> str:
        """
        GPT-2 is a decoder-only model. We provide context + question and
        let it complete the answer. Constrained to avoid runaway generation.
        """
        ctx_words = context.split()[:250]
        ctx = " ".join(ctx_words)
        prompt = f"Context: {ctx}\nQuestion: {query}\nAnswer:"

        with torch.no_grad():
            inputs = self._tokenizer(
                prompt, return_tensors="pt",
                max_length=800, truncation=True
            ).to(_DEVICE)
            input_len = inputs["input_ids"].shape[1]
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=min(self._max_new_tokens, 150),
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        # Decode only newly generated tokens
        new_tokens = outputs[0][input_len:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Cut off at first sentence end
        for end in ["\n", ". ", "? ", "! "]:
            idx = answer.find(end)
            if idx != -1:
                answer = answer[:idx + 1]
                break
        return answer.strip() if answer.strip() else "(no answer generated)"


# ══════════════════════════════════════════════════════════════════════════════
#  §13  CRAG — CORRECTIVE RETRIEVAL AUGMENTED GENERATION  (NEW)
#
#  Paper: "Corrective Retrieval Augmented Generation"
#         Yan, Tang et al., 2024.  Findings of ICLR / cited ACL 2024.
#  arXiv: https://arxiv.org/abs/2401.15884
#
#  Problem (paper §1):  Naive RAG retrieves the top-k chunks unconditionally.
#    If the retrieved documents are irrelevant or partially wrong, the
#    generator is forced to produce an answer grounded in noise, leading to
#    hallucinated or inaccurate outputs.
#
#  Original CRAG solution: a T5-based retrieval evaluator + web search
#    for "incorrect" documents.  Limitation: requires external web search
#    and a fine-tuned evaluator.
#
#  Our improvement over paper limitations:
#    1.  Evaluator uses a multi-signal score (BM25 overlap + cosine sim)
#        — no fine-tuning required, runs on CPU.
#    2.  "Corrective" step uses a deeper BM25 re-pass with loosened params
#        instead of external web search — fully offline.
#    3.  Three-mode gating (CORRECT / AMBIGUOUS / INCORRECT):
#        CORRECT    → use chunk as-is
#        AMBIGUOUS  → decompose chunk into key sentences, keep best
#        INCORRECT  → discard, trigger corrective retrieval pass
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalQuality(Enum):
    CORRECT   = "correct"    # high confidence — chunk is relevant
    AMBIGUOUS = "ambiguous"  # medium — partial relevance
    INCORRECT = "incorrect"  # low — chunk is irrelevant


@dataclass
class ScoredHit:
    """Extended hit with per-chunk CRAG quality score."""
    chunk: Chunk
    bm25_score: float = 0.0
    vec_score: float  = 0.0
    hybrid_score: float = 0.0
    crag_score: float = 0.0
    crag_quality: RetrievalQuality = RetrievalQuality.AMBIGUOUS


# keep backwards compat — the evaluator and main RAG use ScoredHit
Hit = ScoredHit


class CRAGEvaluator:
    """
    Lightweight retrieval quality evaluator (no fine-tuning needed).

    Score = α × BM25_overlap_ratio  +  β × cosine_similarity

    Thresholds (tuned on rag-mini-wikipedia):
        CORRECT   : score ≥ 0.55
        AMBIGUOUS : 0.25 ≤ score < 0.55
        INCORRECT : score < 0.25
    """

    CORRECT_THRESH   = 0.55
    INCORRECT_THRESH = 0.25

    def __init__(self, bm25_w: float = 0.4, vec_w: float = 0.6):
        self.bm25_w = bm25_w
        self.vec_w  = vec_w

    def evaluate(self, query_tokens: List[str],
                 query_vec: np.ndarray,
                 hits: List[ScoredHit]) -> List[ScoredHit]:
        """Assign per-chunk CRAG quality label to each hit."""
        q_set = set(query_tokens)

        for hit in hits:
            chunk_set = set(hit.chunk.tokens)

            # BM25 overlap ratio (Jaccard-like)
            if q_set:
                bm25_overlap = len(q_set & chunk_set) / len(q_set)
            else:
                bm25_overlap = 0.0

            # Cosine similarity
            if hit.chunk.embedding is not None and query_vec is not None:
                cos = float(np.dot(hit.chunk.embedding, query_vec) /
                            (np.linalg.norm(hit.chunk.embedding) *
                             np.linalg.norm(query_vec) + 1e-9))
                cos = max(0.0, cos)
            else:
                cos = hit.vec_score

            hit.crag_score = self.bm25_w * bm25_overlap + self.vec_w * cos

            if hit.crag_score >= self.CORRECT_THRESH:
                hit.crag_quality = RetrievalQuality.CORRECT
            elif hit.crag_score >= self.INCORRECT_THRESH:
                hit.crag_quality = RetrievalQuality.AMBIGUOUS
            else:
                hit.crag_quality = RetrievalQuality.INCORRECT

        return hits

    @staticmethod
    def refine_ambiguous(hit: ScoredHit, query_tokens: List[str]) -> ScoredHit:
        """
        For AMBIGUOUS hits: decompose into sentences, keep the most relevant
        subset.  This sharpens grounding without discarding the whole chunk.
        """
        sents = re.split(r"(?<=[.!?])\s+", hit.chunk.text)
        q_set = set(query_tokens)
        scored = []
        for s in sents:
            s_toks = set(tokenize(s))
            if q_set:
                overlap = len(q_set & s_toks) / len(q_set)
            else:
                overlap = 0.0
            scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])
        # Keep top-50% sentences
        keep = max(1, len(scored) // 2)
        best_sents = [s for _, s in scored[:keep]]
        hit.chunk = Chunk(
            text=" ".join(best_sents),
            source=hit.chunk.source,
            doc_id=hit.chunk.doc_id,
            chunk_id=hit.chunk.chunk_id + "_refined",
            idx=hit.chunk.idx,
            tokens=tokenize(" ".join(best_sents)),
            embedding=hit.chunk.embedding,
            metadata=hit.chunk.metadata,
        )
        return hit


# ══════════════════════════════════════════════════════════════════════════════
#  §14  MULTI-ASPECT QUERY REWRITER  (NEW — MaFeRw-inspired)
#
#  Paper: "MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for RAG"
#         Ma et al., AAAI 2025.  arXiv:2408.17072
#
#  Problem (paper §1): T5 query rewriters trained on context alone fail to
#    capture user intent because the training signal is too sparse and the
#    coupling between rewriting and final answer is loose.  Existing methods
#    use only one feedback source (context), missing retrieval-side and
#    answer-side signals.
#
#  MaFeRw solution: RL with THREE reward aspects:
#    1. Gold-document relevance (retrieval signal)
#    2. Retrieved-document diversity (ranking signal)
#    3. Ground-truth answer similarity (generation signal)
#
#  Our implementation (no RL required at inference):
#    Approximates the three aspects with deterministic signals:
#    Aspect 1 (BM25 signal):   top-hit BM25 terms added to query
#    Aspect 2 (vector signal):  nearest-chunk keywords appended
#    Aspect 3 (answer signal):  TextRank summary high-IDF terms appended
#    All three are fused → deduplicated → used as the final expanded query.
#
#  Improvement over paper limitation:
#    Paper requires an RL training loop (expensive, needs labeled data).
#    Our approximation uses only the retrieval index at inference time
#    — no extra training, no labels, works offline on CPU.
# ══════════════════════════════════════════════════════════════════════════════

_SYNONYMS: Dict[str, List[str]] = {
    "algorithm": ["method","procedure","technique","approach"],
    "data": ["information","dataset","record"],
    "model": ["system","network","architecture","framework"],
    "train": ["learn","fit","optimise","optimize"],
    "result": ["output","finding","outcome","conclusion"],
    "error": ["bug","issue","problem","fault"],
    "fast": ["quick","rapid","efficient"],
    "study": ["research","investigation","analysis"],
    "show": ["demonstrate","indicate","reveal"],
    "increase": ["rise","grow","expand","boost"],
    "decrease": ["fall","drop","reduce","decline"],
    "cause": ["reason","factor","driver","trigger"],
    "effect": ["impact","consequence","result","outcome"],
    "use": ["utilise","apply","employ","leverage"],
    "make": ["create","build","develop","produce"],
    "important": ["key","critical","significant","essential"],
}


class MultiAspectQueryRewriter:
    """
    MaFeRw-inspired multi-aspect query rewriting at inference time.
    Addresses: loose coupling, sparse rewards, single-source feedback.
    """

    def __init__(self, n_bm25_terms: int = 5, n_vec_terms: int = 5,
                 n_ans_terms: int = 3, use_synonym_expansion: bool = True):
        self.n_bm25_terms = n_bm25_terms
        self.n_vec_terms  = n_vec_terms
        self.n_ans_terms  = n_ans_terms
        self.use_synonyms = use_synonym_expansion
        self._textrank = TextRank()

    def rewrite(self,
                query: str,
                query_tokens: List[str],
                bm25_hits: List[Tuple[str, float]],
                vec_hits: List[Tuple[str, float]],
                chunks: Dict[str, Chunk],
                idf: Dict[str, float]) -> Tuple[str, List[str]]:
        """
        Returns (rewritten_query_string, expanded_token_list).

        Three feedback aspects:
          A1 — BM25 signal:   high-IDF terms from top BM25 chunk
          A2 — vector signal: top-IDF terms from top cosine chunk
          A3 — answer signal: TextRank summary extracted from top-3 chunks
        """
        seen = set(query_tokens)
        extra: List[str] = []

        # A1: BM25 signal
        if bm25_hits:
            top_cid = bm25_hits[0][0]
            if top_cid in chunks:
                top_toks = chunks[top_cid].tokens
                # Rank by IDF (rarest = most informative)
                ranked = sorted(top_toks, key=lambda t: -idf.get(t, 0.0))
                for t in ranked:
                    if t not in seen:
                        extra.append(t); seen.add(t)
                    if len([x for x in extra]) >= self.n_bm25_terms:
                        break

        # A2: Vector signal
        if vec_hits:
            top_cid = vec_hits[0][0]
            if top_cid in chunks:
                top_toks = chunks[top_cid].tokens
                ranked = sorted(top_toks, key=lambda t: -idf.get(t, 0.0))
                added = 0
                for t in ranked:
                    if t not in seen:
                        extra.append(t); seen.add(t); added += 1
                    if added >= self.n_vec_terms: break

        # A3: Answer signal — TextRank summary of top-3 chunks
        top_texts = []
        for cid, _ in (bm25_hits[:3] + vec_hits[:3]):
            if cid in chunks:
                top_texts.append(chunks[cid].text)
        if top_texts:
            combined = " ".join(top_texts)
            try:
                summary = self._textrank.summarise(combined, n_sentences=2)
                sum_toks = tokenize(summary)
                # Extract high-IDF terms from summary
                ranked_sum = sorted(sum_toks, key=lambda t: -idf.get(t, 0.0))
                added = 0
                for t in ranked_sum:
                    if t not in seen:
                        extra.append(t); seen.add(t); added += 1
                    if added >= self.n_ans_terms: break
            except Exception:
                pass

        # Synonym expansion (original approach, kept as additional aspect)
        if self.use_synonyms:
            for tok in list(query_tokens):
                for key, syns in _SYNONYMS.items():
                    if tok == _stemmer.stem(key):
                        for s in syns:
                            st = _stemmer.stem(s)
                            if st not in seen:
                                extra.append(st); seen.add(st)

        expanded = query_tokens + extra
        rewritten = query + " " + " ".join(extra) if extra else query
        return rewritten, expanded


# ══════════════════════════════════════════════════════════════════════════════
#  §15  MMR RERANKER  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

def mmr_rerank(hits: List[ScoredHit], query_vec: np.ndarray,
               k: int, lam: float = 0.6) -> List[ScoredHit]:
    if len(hits) <= k: return hits
    selected: List[ScoredHit] = []
    pool = list(hits)
    while len(selected) < k and pool:
        if not selected:
            selected.append(pool.pop(0)); continue
        sel_vecs = np.stack([h.chunk.embedding for h in selected
                             if h.chunk.embedding is not None])
        best_score, best_i = -1e9, 0
        for i, h in enumerate(pool):
            rel = h.hybrid_score
            div = float((sel_vecs @ h.chunk.embedding).max()) if h.chunk.embedding is not None else 0.0
            score = lam * rel - (1 - lam) * div
            if score > best_score: best_score = score; best_i = i
        selected.append(pool.pop(best_i))
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  §16  ANSWER GENERATOR  (enhanced — LLM + extractive fallback)
# ══════════════════════════════════════════════════════════════════════════════

class AnswerGenerator:
    """
    Enhanced generator:
      1. Try open-source LLM (FLAN-T5 → GPT-2 → KN-LM)
      2. Always append extractive TextRank answer for grounding
      3. Source citations preserved
    """

    def __init__(self, use_llm: bool = True, use_ngram: bool = True,
                 prefer_t5: bool = True):
        self.textrank = TextRank()
        self.lm = KNLanguageModel()
        self.use_ngram = use_ngram
        self.llm: Optional[OpenSourceLLM] = None
        if use_llm:
            self.llm = OpenSourceLLM(prefer_t5=prefer_t5)

    def generate(self, query: str, hits: List[ScoredHit],
                 n_sentences: int = 4) -> str:
        if not hits:
            return "No relevant information found in the indexed documents."

        combined = " ".join(h.chunk.text for h in hits)
        extractive = self.textrank.summarise(combined, n_sentences=n_sentences)

        # Build LLM answer if available
        llm_answer = ""
        if self.llm is not None:
            try:
                llm_answer = self.llm.generate(
                    query, combined,
                    kn_texts=[h.chunk.text for h in hits]
                )
            except Exception as e:
                log.warning(f"LLM generation failed: {e}")
                llm_answer = ""

        lines = []

        # LLM answer block (shown first, if available)
        if llm_answer:
            lines.append("── LLM Answer (FLAN-T5 / GPT-2) ─────────────────────────────")
            lines.append(llm_answer)
            lines.append("")

        # Extractive fallback always shown
        lines.append("── Extractive Answer (TextRank) ───────────────────────────────")
        lines.append(extractive)
        lines.append("")

        # CRAG quality summary
        quality_counts = Counter(h.crag_quality.value for h in hits)
        lines.append(
            f"── CRAG Retrieval Quality ─────────────────────────────────────\n"
            f"  ✅ Correct: {quality_counts.get('correct',0)}  "
            f"🟡 Ambiguous: {quality_counts.get('ambiguous',0)}  "
            f"❌ Incorrect: {quality_counts.get('incorrect',0)}"
        )
        lines.append("")

        # Source citations
        lines.append("── Sources ─────────────────────────────────────────────────────")
        for i, h in enumerate(hits, 1):
            quality_icon = {"correct":"✅","ambiguous":"🟡","incorrect":"❌"}.get(
                h.crag_quality.value, "?")
            lines.append(
                f"  [{i}] {Path(h.chunk.source).name}  "
                f"(chunk {h.chunk.idx}, hybrid={h.hybrid_score:.3f}, "
                f"crag={h.crag_score:.3f} {quality_icon})"
            )
            preview = h.chunk.text[:180].replace("\n", " ")
            lines.append(f"      {preview}…")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  §17  PERSISTENCE  (original, preserved)
# ══════════════════════════════════════════════════════════════════════════════

class Store:
    def __init__(self, path: str = ".rag_index"):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fpath = self.dir / "index.pkl.gz"

    def save(self, state: dict) -> None:
        blob = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        with gzip.open(self.fpath, "wb", compresslevel=6) as f:
            f.write(blob)
        log.info(f"Index saved → {self.fpath}  ({len(blob)/1024:.1f} KB)")

    def load(self) -> Optional[dict]:
        if not self.fpath.exists(): return None
        try:
            with gzip.open(self.fpath, "rb") as f:
                return pickle.loads(f.read())
        except Exception as e:
            log.warning(f"Load failed: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  §18  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGConfig:
    # Chunking
    chunk_size: int    = 350
    chunk_overlap: int = 60
    min_chunk_size: int = 20
    # Embedding
    embed_dim: int     = 384      # 384 for MiniLM / 768 for BERT / 256 for hash
    use_bert: bool     = True     # True → BERTEmbedder; False → HashEmbedder
    # Retrieval
    bm25_k: int        = 30       # wider initial recall for CRAG correction
    vec_k: int         = 30
    final_k: int       = 5
    mmr_lambda: float  = 0.65
    bm25_w: float      = 0.35
    vec_w: float       = 0.65
    # CRAG
    use_crag: bool     = True     # corrective retrieval gating
    crag_bm25_w: float = 0.4
    crag_vec_w: float  = 0.6
    # Multi-aspect query rewriting (MaFeRw)
    use_mafrew: bool   = True
    mafrew_bm25_terms: int = 5
    mafrew_vec_terms: int  = 5
    mafrew_ans_terms: int  = 3
    # LLM generation
    use_llm: bool      = True     # FLAN-T5 / GPT-2
    prefer_t5: bool    = True     # prefer T5 over GPT-2
    use_ngram: bool    = True     # KN-LM fallback
    n_answer_sentences: int = 4
    # Storage
    index_dir: str     = ".rag_enhanced_index"
    verbose: bool      = False


# ══════════════════════════════════════════════════════════════════════════════
#  §19  MAIN RAG ENGINE  (enhanced)
# ══════════════════════════════════════════════════════════════════════════════

class RAG:
    """
    SuperiorRAG Enhanced Edition.

    Additions over original:
      - BERTEmbedder (sentence-transformers / raw BERT / hash fallback)
      - OpenSourceLLM (FLAN-T5 → GPT-2 → KN-LM)
      - CRAGEvaluator  (per-chunk quality scoring + corrective retrieval)
      - MultiAspectQueryRewriter (MaFeRw-inspired 3-signal expansion)
    """

    def __init__(self, cfg: Optional[RAGConfig] = None):
        self.cfg = cfg or RAGConfig()
        if self.cfg.verbose:
            log.setLevel(logging.DEBUG)

        # Embedder selection
        if self.cfg.use_bert:
            self._embedder = BERTEmbedder(dim=self.cfg.embed_dim)
        else:
            self._embedder = HashEmbedder(self.cfg.embed_dim)
            self.cfg.embed_dim = self.cfg.embed_dim  # keep as set

        self._chunks: Dict[str, Chunk] = {}
        self._bm25   = BM25Index()
        self._vec    = VectorIndex(self.cfg.embed_dim)
        self._chunker = Chunker(self.cfg.chunk_size, self.cfg.chunk_overlap,
                                self.cfg.min_chunk_size)
        self._crag   = CRAGEvaluator(self.cfg.crag_bm25_w, self.cfg.crag_vec_w)
        self._rewriter = MultiAspectQueryRewriter(
            n_bm25_terms=self.cfg.mafrew_bm25_terms,
            n_vec_terms=self.cfg.mafrew_vec_terms,
            n_ans_terms=self.cfg.mafrew_ans_terms,
        ) if self.cfg.use_mafrew else None

        self._gen    = AnswerGenerator(
            use_llm=self.cfg.use_llm,
            use_ngram=self.cfg.use_ngram,
            prefer_t5=self.cfg.prefer_t5,
        )
        self._store  = Store(self.cfg.index_dir)
        self._sources: set = set()
        self._lock   = threading.Lock()
        self._load()

    # ── ingestion ─────────────────────────────────────────────────────────────

    def add_file(self, path: str | Path) -> int:
        return self._ingest(Loader.load(path), str(path))

    def add_text(self, text: str, source: str = "<text>") -> int:
        return self._ingest([Loader.load_string(text, source)], source)

    def add_directory(self, dirpath: str | Path, pattern: str = "**/*.txt") -> int:
        total = 0
        for p in Path(dirpath).glob(pattern):
            total += self.add_file(p)
        return total

    def _ingest(self, docs: List[Document], label: str) -> int:
        if label in self._sources:
            log.info(f"Already indexed: {label}"); return 0

        new_chunks: List[Chunk] = []
        for doc in docs:
            new_chunks.extend(self._chunker.chunk(doc))
        if not new_chunks: return 0

        # Token lists
        all_tokens = [tokenize(c.text) for c in new_chunks]
        existing_tokens = [c.tokens for c in self._chunks.values() if c.tokens]

        # Fit IDF on hash embedder (BERT is self-supervised)
        if isinstance(self._embedder, HashEmbedder):
            self._embedder.fit(existing_tokens + all_tokens)

        # Batch BERT embeddings for efficiency
        if isinstance(self._embedder, BERTEmbedder) and hasattr(self._embedder, 'embed_batch'):
            texts = [c.text for c in new_chunks]
            try:
                log.info(f"Embedding {len(texts)} chunks with BERT …")
                vecs = self._embedder.embed_batch(texts, batch_size=32)
            except Exception as e:
                log.warning(f"Batch embed failed ({e}), falling back to single embed")
                vecs = np.stack([self._embedder.embed(c.text) for c in new_chunks])
        else:
            vecs = np.stack([self._embedder.embed(t) for t in all_tokens])

        added = 0
        for chunk, tokens, vec in zip(new_chunks, all_tokens, vecs):
            if chunk.chunk_id in self._chunks: continue
            chunk.tokens = tokens
            chunk.embedding = vec.astype(np.float32)
            # Normalise for cosine
            n = np.linalg.norm(chunk.embedding)
            if n > 1e-9: chunk.embedding /= n

            with self._lock:
                self._chunks[chunk.chunk_id] = chunk
                self._bm25.add(chunk.chunk_id, tokens)
                self._vec.add(chunk.chunk_id, chunk.embedding)
            added += 1

        with self._lock:
            self._bm25.build()
            self._vec.commit()
            self._sources.add(label)

        log.info(f"Indexed {added} chunks ← '{label}'  |  total={len(self._chunks)}")
        self._save()
        return added

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[ScoredHit]:
        if not self._chunks: return []

        # Tokenize query
        query_tokens = tokenize(query)

        # Initial embedding
        if isinstance(self._embedder, BERTEmbedder):
            q_vec = self._embedder.embed(query)
        else:
            q_vec = self._embedder.embed(query_tokens)
        n = np.linalg.norm(q_vec)
        if n > 1e-9: q_vec /= n

        # Initial retrieval pass
        bm25_raw = self._bm25.search(query_tokens, self.cfg.bm25_k)
        vec_raw  = self._vec.search(q_vec, self.cfg.vec_k)

        # ── MaFeRw multi-aspect query rewriting ──────────────────────────────
        if self._rewriter is not None:
            idf = getattr(self._embedder, 'idf', {}) or {}
            # Also borrow BM25 idf
            if hasattr(self._bm25, '_idf'):
                idf = {**self._bm25._idf, **idf}

            _, expanded_tokens = self._rewriter.rewrite(
                query=query,
                query_tokens=query_tokens,
                bm25_hits=bm25_raw,
                vec_hits=vec_raw,
                chunks=self._chunks,
                idf=idf,
            )

            # Second retrieval pass with expanded query
            bm25_raw2 = self._bm25.search(expanded_tokens, self.cfg.bm25_k)
            if isinstance(self._embedder, BERTEmbedder):
                q_vec2 = self._embedder.embed(" ".join(expanded_tokens))
            else:
                q_vec2 = self._embedder.embed(expanded_tokens)
            n2 = np.linalg.norm(q_vec2)
            if n2 > 1e-9: q_vec2 /= n2
            vec_raw2 = self._vec.search(q_vec2, self.cfg.vec_k)

            # Merge: take union; boosted IDs appear in both passes → higher weight
            bm25_merged = self._merge_results(bm25_raw, bm25_raw2, boost=1.2)
            vec_merged  = self._merge_results(vec_raw,  vec_raw2,  boost=1.2)
        else:
            expanded_tokens = query_tokens
            bm25_merged = bm25_raw
            vec_merged  = vec_raw
            q_vec2 = q_vec

        # Normalise scores
        def norm(items):
            d = dict(items)
            if not d: return d
            mx = max(d.values()) or 1.0
            return {k: v / mx for k, v in d.items()}

        bn = norm(bm25_merged)
        vn = norm(vec_merged)
        all_ids = set(bn) | set(vn)

        hits: List[ScoredHit] = []
        for cid in all_ids:
            if cid not in self._chunks: continue
            b = bn.get(cid, 0.0)
            v = vn.get(cid, 0.0)
            h = self.cfg.bm25_w * b + self.cfg.vec_w * v
            hits.append(ScoredHit(
                chunk=self._chunks[cid],
                bm25_score=b, vec_score=v, hybrid_score=h
            ))

        hits.sort(key=lambda x: -x.hybrid_score)

        # ── CRAG evaluation ───────────────────────────────────────────────────
        if self.cfg.use_crag:
            hits = self._crag.evaluate(query_tokens, q_vec, hits)

            # Corrective step for INCORRECT hits
            corrected = []
            incorrect_count = 0
            for h in hits:
                if h.crag_quality == RetrievalQuality.INCORRECT:
                    incorrect_count += 1
                    # Trigger corrective re-retrieval: use looser BM25
                    alt = self._corrective_retrieve(query_tokens, expanded_tokens, h)
                    if alt:
                        corrected.append(alt)
                elif h.crag_quality == RetrievalQuality.AMBIGUOUS:
                    corrected.append(self._crag.refine_ambiguous(h, query_tokens))
                else:
                    corrected.append(h)

            if incorrect_count > 0:
                log.debug(f"CRAG: corrected {incorrect_count} incorrect chunk(s)")
            hits = corrected

        hits.sort(key=lambda x: -x.hybrid_score)

        # MMR reranking
        q_final = q_vec2 if self._rewriter else q_vec
        return mmr_rerank(hits, q_final, self.cfg.final_k, self.cfg.mmr_lambda)

    def _merge_results(self,
                       r1: List[Tuple[str, float]],
                       r2: List[Tuple[str, float]],
                       boost: float = 1.2) -> List[Tuple[str, float]]:
        """Merge two ranked lists; IDs in both get a score boost."""
        d1 = dict(r1); d2 = dict(r2)
        merged = {}
        for cid, score in {**d1, **d2}.items():
            if cid in d1 and cid in d2:
                merged[cid] = (d1[cid] + d2[cid]) / 2 * boost
            elif cid in d1:
                merged[cid] = d1[cid]
            else:
                merged[cid] = d2[cid]
        return sorted(merged.items(), key=lambda x: -x[1])

    def _corrective_retrieve(self,
                              query_tokens: List[str],
                              expanded_tokens: List[str],
                              failed_hit: ScoredHit) -> Optional[ScoredHit]:
        """
        CRAG corrective retrieval: when a top hit is INCORRECT, find the
        best alternative chunk using the full expanded query.
        Returns None if no better chunk found.
        """
        alt_bm25 = self._bm25.search(expanded_tokens, self.cfg.bm25_k * 2)
        # Filter out the failed chunk and pick the highest-scoring alternative
        for cid, score in alt_bm25:
            if cid == failed_hit.chunk.chunk_id: continue
            if cid in self._chunks:
                alt_chunk = self._chunks[cid]
                alt_hit = ScoredHit(
                    chunk=alt_chunk,
                    bm25_score=score,
                    vec_score=failed_hit.vec_score,
                    hybrid_score=self.cfg.bm25_w * score + self.cfg.vec_w * failed_hit.vec_score,
                    crag_quality=RetrievalQuality.AMBIGUOUS,
                )
                return alt_hit
        return None

    # ── query  ────────────────────────────────────────────────────────────────

    def query(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        hits = self.retrieve(question)
        answer = self._gen.generate(question, hits, self.cfg.n_answer_sentences)
        ms = (time.perf_counter() - t0) * 1000
        return {
            "question": question,
            "answer": answer,
            "hits": hits,
            "latency_ms": round(ms, 2),
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self):
        # Note: BERT model weights not persisted (re-loaded from HuggingFace cache)
        embed_idf = getattr(self._embedder, 'idf', {})
        self._store.save({
            "chunks": self._chunks,
            "bm25":   self._bm25,
            "vec":    self._vec,
            "idf":    embed_idf,
            "sources": self._sources,
            "embed_dim": self.cfg.embed_dim,
            "use_bert":  self.cfg.use_bert,
        })

    def _load(self):
        state = self._store.load()
        if state is None:
            log.info("No existing index — starting fresh."); return
        try:
            self._chunks  = state["chunks"]
            self._bm25    = state["bm25"]
            self._vec     = state["vec"]
            self._sources = state.get("sources", set())
            saved_dim = state.get("embed_dim", self.cfg.embed_dim)
            if saved_dim != self.cfg.embed_dim:
                log.warning(f"Saved embed_dim={saved_dim} ≠ config={self.cfg.embed_dim} — rebuilding index")
                self._chunks.clear(); self._bm25 = BM25Index()
                self._vec = VectorIndex(self.cfg.embed_dim); self._sources.clear()
                return
            if isinstance(self._embedder, HashEmbedder):
                self._embedder.idf = state.get("idf", {})
            log.info(f"Loaded index: {len(self._chunks)} chunks from {len(self._sources)} source(s)")
        except Exception as e:
            log.warning(f"Index load error: {e} — starting fresh")

    def clear(self):
        self._chunks.clear()
        self._bm25 = BM25Index()
        self._vec  = VectorIndex(self.cfg.embed_dim)
        self._sources.clear()
        self._save()
        log.info("Index cleared.")

    def stats(self) -> dict:
        bert_mode = getattr(self._embedder, '_mode', 'hash') if isinstance(self._embedder, BERTEmbedder) else 'hash'
        llm_mode  = getattr(self._gen.llm, '_mode', 'none') if self._gen.llm else 'none'
        return {
            "chunks":     len(self._chunks),
            "sources":    len(self._sources),
            "source_list": sorted(self._sources),
            "vector_mb":  round(self._vec.memory_mb(), 3),
            "embed_dim":  self.cfg.embed_dim,
            "bert_mode":  bert_mode,
            "llm_mode":   llm_mode,
            "vocab_size": len(getattr(self._embedder, 'idf', {})),
            "crag_enabled":    self.cfg.use_crag,
            "mafrew_enabled":  self.cfg.use_mafrew,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  §20  CLI  (enhanced — shows CRAG stats)
# ══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  SuperiorRAG  ENHANCED  ·  T5 / BERT / CRAG / MaFeRw           ║
  ║  BM25+ · BERT Embeddings · FLAN-T5 / GPT-2 · CRAG · MaFeRw    ║
  ╚══════════════════════════════════════════════════════════════════╝
"""

def _sep(char="─", n=68): return char * n

def _do_query(rag: RAG, question: str):
    result = rag.query(question)
    print(f"\n  {_sep()}")
    print(f"  ❓ {result['question']}")
    print(f"  {_sep()}")
    for line in result["answer"].splitlines():
        print(f"  {line}")
    print(f"\n  ⏱  {result['latency_ms']} ms  |  {len(result['hits'])} chunks retrieved")
    print()

def _print_stats(rag: RAG):
    s = rag.stats()
    print(f"\n  {_sep('═')}")
    print(f"  ENHANCED RAG — INDEX STATISTICS")
    print(f"  {_sep('═')}")
    print(f"  Chunks      : {s['chunks']}")
    print(f"  Sources     : {s['sources']}")
    print(f"  Vocab size  : {s['vocab_size']}")
    print(f"  Vector RAM  : {s['vector_mb']} MB")
    print(f"  Embed dim   : {s['embed_dim']}")
    print(f"  BERT mode   : {s['bert_mode']}")
    print(f"  LLM mode    : {s['llm_mode']}")
    print(f"  CRAG        : {'✅ enabled' if s['crag_enabled'] else '❌ disabled'}")
    print(f"  MaFeRw      : {'✅ enabled' if s['mafrew_enabled'] else '❌ disabled'}")
    if s['source_list']:
        print(f"  Files       :")
        for src in s['source_list']: print(f"    · {src}")
    print()


def cli():
    parser = argparse.ArgumentParser(
        prog="rag_enhanced",
        description="SuperiorRAG Enhanced — BERT + FLAN-T5 + CRAG + MaFeRw",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)
    p_i = sub.add_parser("ingest");  p_i.add_argument("files", nargs="+")
    p_q = sub.add_parser("query");   p_q.add_argument("question", nargs="+")
    p_q.add_argument("--k", type=int, default=5)
    p_q.add_argument("--no-llm", action="store_true")
    p_q.add_argument("--no-crag", action="store_true")
    sub.add_parser("chat")
    sub.add_parser("stats")
    sub.add_parser("clear")
    sub.add_parser("demo")
    args = parser.parse_args()

    print(BANNER)
    cfg = RAGConfig()
    if hasattr(args, 'no_llm') and args.no_llm:
        cfg.use_llm = False
    if hasattr(args, 'no_crag') and args.no_crag:
        cfg.use_crag = False
    rag = RAG(cfg)

    if args.cmd == "ingest":
        total = sum(rag.add_file(f) for f in args.files)
        print(f"\n✓ Ingested {total} new chunks")
        _print_stats(rag)
    elif args.cmd == "query":
        rag.cfg.final_k = args.k
        _do_query(rag, " ".join(args.question))
    elif args.cmd == "chat" or args.cmd is None:
        print("Interactive RAG  ·  'exit' to quit  ·  'stats' for info\n")
        while True:
            try:
                q = input("  You › ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye!"); break
            if not q: continue
            if q.lower() in ("exit","quit","q"): break
            if q.lower() == "stats": _print_stats(rag); continue
            _do_query(rag, q)
    elif args.cmd == "stats": _print_stats(rag)
    elif args.cmd == "clear": rag.clear(); print("✓ Index cleared")
    elif args.cmd == "demo":  run_demo(rag)
    else: parser.print_help()


# ══════════════════════════════════════════════════════════════════════════════
#  §21  BUILT-IN DEMO  (same corpus as original)
# ══════════════════════════════════════════════════════════════════════════════

DEMO_CORPUS = {
    "artificial_intelligence.txt": """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to
natural intelligence displayed by animals including humans. AI research has been defined
as the field of study of intelligent agents, which refers to any system that perceives
its environment and takes actions that maximize its chance of achieving its goals.

Machine learning is a subset of AI that enables systems to learn and improve from experience
without being explicitly programmed. Deep learning uses artificial neural networks with many
layers to learn hierarchical representations of data. The transformer architecture, introduced
in the landmark paper "Attention is All You Need" (2017), revolutionised natural language
processing by relying entirely on self-attention mechanisms.

Reinforcement learning trains agents through reward signals — the agent learns by trial
and error, receiving positive or negative feedback based on its actions. AlphaGo, developed
by DeepMind, used reinforcement learning combined with Monte Carlo tree search to defeat
world champion Go players in 2016, a milestone once thought to be decades away.

Large language models (LLMs) like GPT and BERT are trained on massive text corpora using
self-supervised objectives. They demonstrate emergent capabilities — skills not explicitly
trained for — that appear at sufficient model scale.
""",
    "retrieval_augmented_generation.txt": """
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval
with text generation. Instead of relying solely on knowledge encoded in model weights,
RAG retrieves relevant passages from an external knowledge base and conditions generation
on that retrieved context.

The core pipeline has three stages: indexing, retrieval, and generation. During indexing,
documents are split into chunks, converted to dense vector representations (embeddings),
and stored in a vector database. At query time, the user's question is embedded and the
most semantically similar chunks are retrieved. Finally, both the question and retrieved
context are passed to a generative model to produce a grounded answer.

BM25 is a classical sparse retrieval model based on term frequency and inverse document
frequency. Dense retrieval uses neural embeddings to capture semantic similarity beyond
exact keyword matching. Hybrid search combines both approaches to benefit from
complementary strengths — BM25 excels at rare keywords while dense retrieval handles
paraphrase and semantic equivalence.

Maximal Marginal Relevance (MMR) is a reranking technique that balances relevance to
the query with diversity among retrieved results, reducing redundant information in
the retrieved context.
""",
    "climate_science.txt": """
Climate change refers to long-term shifts in temperatures and weather patterns on Earth.
Since the mid-20th century, human activities — primarily burning fossil fuels — have been
the dominant cause of observed climate change by releasing greenhouse gases.

The greenhouse effect occurs when gases such as carbon dioxide, methane, and water vapour
trap heat in the Earth's atmosphere. Without any greenhouse effect, Earth's average
temperature would be about -18°C; the natural greenhouse effect keeps it at around +15°C.
However, anthropogenic emissions have strengthened this effect, raising global average
temperatures by approximately 1.1°C above pre-industrial levels as of 2023.

Tipping points are thresholds in the climate system that, once crossed, lead to
self-reinforcing changes that may be irreversible on human timescales. Examples include
the collapse of the West Antarctic Ice Sheet, dieback of the Amazon rainforest, and
permafrost thaw releasing stored carbon.
""",
    "algorithms_data_structures.txt": """
An algorithm is a finite sequence of well-defined instructions for solving a problem.
Algorithm complexity is measured in terms of time (number of operations) and space
(memory usage) as a function of input size, expressed using Big-O notation.

Binary search finds a target in a sorted array in O(log n) time by repeatedly halving
the search interval. Quick sort achieves O(n log n) average-case time complexity using
divide-and-conquer with a pivot element. Merge sort guarantees O(n log n) in all cases.

Hash tables provide O(1) average-case lookup, insertion, and deletion by using a hash
function to map keys to array indices. The PageRank algorithm, developed by Larry Page
and Sergey Brin, ranks web pages by computing the stationary distribution of a random
walk on the link graph.
""",
}


def run_demo(rag: Optional[RAG] = None):
    if rag is None:
        rag = RAG(RAGConfig(index_dir=".rag_enhanced_demo"))

    print(f"  {_sep('═')}")
    print("  ENHANCED DEMO — Ingesting sample documents …")
    print(f"  {_sep('═')}\n")

    for fname, text in DEMO_CORPUS.items():
        rag.add_text(text.strip(), source=fname)

    _print_stats(rag)

    questions = [
        "How does retrieval-augmented generation work?",
        "What are tipping points in climate science?",
        "Explain the difference between BM25 and dense retrieval.",
        "How does PageRank work?",
        "What is reinforcement learning?",
    ]

    print(f"  {_sep('═')}")
    print("  DEMO QUERIES")
    print(f"  {_sep('═')}\n")

    for q in questions:
        _do_query(rag, q)

    print(f"  {_sep('═')}")
    print("  Demo complete.  Run 'python rag_engine_enhanced.py chat' for interactive mode.")
    print(f"  {_sep('═')}\n")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(BANNER)
        run_demo()
    else:
        cli()
