"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   SuperiorRAG  ·  WEB  EDITION                                               ║
║                                                                              ║
║   ARCHITECTURE:                                                              ║
║   ┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────┐  ║
║   │  Query      │───▶│  Web Scraper │───▶│  ChromaDB     │───▶│ DeepSeek │  ║
║   │  (user)     │    │  (live data) │    │  Vector Store │    │ (Ollama) │  ║
║   └─────────────┘    └──────────────┘    └───────────────┘    └──────────┘  ║
║                                                                              ║
║   INSTALL:                                                                   ║
║     pip install requests beautifulsoup4 chromadb sentence-transformers       ║
║     pip install transformers torch                                           ║
║     # For DeepSeek (recommended, free, local):                               ║
║     #   1. Download Ollama from https://ollama.com                           ║
║     #   2. Run: ollama pull deepseek-r1:7b                                   ║
║     #   3. Run: ollama serve        (keep this running)                      ║
║                                                                              ║
║   RUN:                                                                       ║
║     python rag_web.py                          # interactive chat            ║
║     python rag_web.py --query "What is RAG?"  # single query                ║
║     python rag_web.py --sources 5             # scrape 5 pages              ║
║     python rag_web.py --model llama3.2        # use a different model       ║
║     python rag_web.py --clear                 # wipe ChromaDB               ║
║     python rag_web.py --stats                 # show system info            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import re, os, sys, json, math, time, hashlib, logging
import argparse, textwrap, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, quote_plus

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RAG-WEB")

# ── optional imports ──────────────────────────────────────────────────────────
try:
    import requests
    from bs4 import BeautifulSoup
    _HAS_SCRAPER = True
except ImportError:
    _HAS_SCRAPER = False
    log.warning("pip install requests beautifulsoup4")

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False
    log.info("chromadb not found — using numpy in-memory fallback")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import torch
    from transformers import (
        T5ForConditionalGeneration, T5Tokenizer,
        GPT2LMHeadModel, GPT2Tokenizer,
    )
    _HAS_TORCH = True
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TORCH = False
    _HAS_TRANSFORMERS = False

_DEVICE = "cpu"
if _HAS_TORCH:
    if torch.cuda.is_available():
        _DEVICE = "cuda"
    else:
        try:
            if torch.backends.mps.is_available():
                _DEVICE = "mps"
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  §1  TOKENIZER  (zero-dep)
# ══════════════════════════════════════════════════════════════════════════════

_STOPS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can cannot could did do does doing
don down during each even few for from further get got had has have having he
her here hers herself him himself his how however i if in into is it its itself
just let me more most my myself new no nor not now of off on once only or other
our out over own re same she should since so some such than that the their them
themselves then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your
yours yourself yourselves
""".split())

_WRE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*\b")


class PorterStemmer:
    _V = frozenset("aeiou")

    def stem(self, w):
        w = w.lower()
        if len(w) <= 2: return w
        for fn in [self._1a, self._1b, self._1c, self._2, self._3, self._4, self._5a, self._5b]:
            w = fn(w)
        return w

    def _c(self, w, i):
        c = w[i]
        if c in self._V: return False
        if c == "y": return i == 0 or not self._c(w, i-1)
        return True

    def _m(self, w):
        n, i, m = len(w), 0, 0
        while i < n and self._c(w, i): i += 1
        while i < n:
            while i < n and not self._c(w, i): i += 1
            m += 1
            while i < n and self._c(w, i): i += 1
        return m

    def _hv(self, s): return any(not self._c(s, i) for i in range(len(s)))
    def _dc(self, w): return len(w) >= 2 and w[-1] == w[-2] and self._c(w, len(w)-1)
    def _cvc(self, w):
        n = len(w)
        return n >= 3 and self._c(w, n-3) and not self._c(w, n-2) and self._c(w, n-1) and w[-1] not in "wxy"

    def _1a(self, w):
        if w.endswith("sses"): return w[:-2]
        if w.endswith("ies"): return w[:-2]
        if w.endswith("ss"): return w
        if w.endswith("s"): return w[:-1]
        return w

    def _1b(self, w):
        if w.endswith("eed"):
            if self._m(w[:-3]) > 0: return w[:-1]
            return w
        flag = False
        for s in ("ed", "ing"):
            if w.endswith(s) and self._hv(w[:-len(s)]):
                w = w[:-len(s)]; flag = True; break
        if flag:
            for s2, r in [("at", "ate"), ("bl", "ble"), ("iz", "ize")]:
                if w.endswith(s2): return w + r[2:]
            if self._dc(w) and w[-1] not in "lsz": return w[:-1]
            if self._m(w) == 1 and self._cvc(w): return w + "e"
        return w

    def _1c(self, w):
        if w.endswith("y") and self._hv(w[:-1]): return w[:-1] + "i"
        return w

    def _2(self, w):
        for s, r in [("ational","ate"),("tional","tion"),("enci","ence"),("anci","ance"),
                     ("izer","ize"),("abli","able"),("alli","al"),("entli","ent"),
                     ("eli","e"),("ousli","ous"),("ization","ize"),("ation","ate"),
                     ("ator","ate"),("alism","al"),("iveness","ive"),("fulness","ful"),
                     ("ousness","ous"),("aliti","al"),("iviti","ive"),("biliti","ble")]:
            if w.endswith(s) and self._m(w[:-len(s)]) > 0: return w[:-len(s)] + r
        return w

    def _3(self, w):
        for s, r in [("icate","ic"),("ative",""),("alize","al"),("iciti","ic"),
                     ("ical","ic"),("ful",""),("ness","")]:
            if w.endswith(s) and self._m(w[:-len(s)]) > 0: return w[:-len(s)] + r
        return w

    def _4(self, w):
        for s in ["al","ance","ence","er","ic","able","ible","ant","ement","ment",
                  "ent","ion","ou","ism","ate","iti","ous","ive","ize"]:
            if w.endswith(s):
                stem = w[:-len(s)]
                if s == "ion":
                    if self._m(stem) > 1 and stem and stem[-1] in "st": return stem
                elif self._m(stem) > 1: return stem
        return w

    def _5a(self, w):
        if w.endswith("e"):
            s = w[:-1]; m = self._m(s)
            if m > 1 or (m == 1 and not self._cvc(s)): return s
        return w

    def _5b(self, w):
        if self._m(w) > 1 and self._dc(w) and w.endswith("l"): return w[:-1]
        return w


_PS = PorterStemmer()


def tokenize(text: str, stem: bool = True, stops: bool = True) -> List[str]:
    words = [w.lower() for w in _WRE.findall(text)]
    if stops: words = [w for w in words if w not in _STOPS and len(w) > 1]
    if stem:  words = [_PS.stem(w) for w in words]
    return words


# ══════════════════════════════════════════════════════════════════════════════
#  §2  WEB SCRAPER
# ══════════════════════════════════════════════════════════════════════════════

_SKIP_DOMAINS = frozenset([
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "linkedin.com", "pinterest.com", "youtube.com", "amazon.com", "ebay.com",
])

# Rotate through realistic browser User-Agents to avoid blocks
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

import random as _random

def _make_headers(ua: Optional[str] = None) -> dict:
    return {
        "User-Agent": ua or _random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }


@dataclass
class ScrapedPage:
    url:     str
    title:   str
    content: str
    source:  str


class WebScraper:
    """
    Robust web scraper with 5 search strategies and anti-block measures.

    Search strategy order (tries each until URLs are found):
      1. DuckDuckGo Instant Answer API  (JSON, most reliable)
      2. DuckDuckGo HTML                (with retries + rotating UA)
      3. Google via opensearch / lucky  (direct URL construction)
      4. Bing HTML                      (with retries)
      5. Wikipedia API                  (fallback for factual queries)
      6. Direct known-good URLs         (hardcoded for common topics)
    """

    # Public SearXNG instances (community-run, no key needed)
    _SEARXNG_INSTANCES = [
        "https://searx.be/search",
        "https://search.bus-hit.me/search",
        "https://searxng.site/search",
        "https://paulgo.io/search",
    ]

    def __init__(self, max_pages=5, timeout=12, max_content_len=6000):
        self.max_pages       = max_pages
        self.timeout         = timeout
        self.max_content_len = max_content_len

    # ── main entry ────────────────────────────────────────────────────────────

    def scrape_query(self, query: str) -> List[ScrapedPage]:
        if not _HAS_SCRAPER:
            raise RuntimeError("pip install requests beautifulsoup4")

        log.info(f"🔍 Searching: '{query}'")

        # Try each search strategy in order
        urls: List[str] = []
        strategies = [
            ("DDG Instant API",  self._search_ddg_api),
            ("DDG HTML",         self._search_ddg_html),
            ("SearXNG",          self._search_searxng),
            ("Bing",             self._search_bing),
            ("Google Scholar",   self._search_google),
            ("Wikipedia API",    self._search_wikipedia_api),
        ]

        for name, fn in strategies:
            try:
                found = fn(query)
                if found:
                    log.info(f"  🔎 {name} → {len(found)} URLs")
                    urls.extend(found)
                    if len(urls) >= self.max_pages * 2:
                        break
                else:
                    log.debug(f"  ⏭  {name} returned nothing")
            except Exception as e:
                log.debug(f"  ⚠  {name} error: {e}")

        # De-duplicate while preserving order
        seen = set()
        unique_urls = []
        for u in urls:
            if u not in seen and not self._skip(u):
                seen.add(u); unique_urls.append(u)

        if not unique_urls:
            log.warning("All search strategies failed — trying Wikipedia directly.")
            wiki_pages = self._wikipedia_direct(query)
            if wiki_pages:
                return wiki_pages
            log.error("❌ No URLs found. Check internet connection.")
            return []

        # Fetch pages
        pages: List[ScrapedPage] = []
        for url in unique_urls:
            if len(pages) >= self.max_pages:
                break
            page = self._fetch(url)
            if page and len(page.content.split()) > 50:
                pages.append(page)
                log.info(f"  ✅ [{len(pages)}] {page.source} — {page.title[:55]}")
            else:
                log.debug(f"  ⏭  Too short / failed: {url}")

        log.info(f"📄 Scraped {len(pages)} pages for '{query}'")
        return pages

    # ── strategy 1: DuckDuckGo Instant Answer JSON API ────────────────────────

    def _search_ddg_api(self, query: str) -> List[str]:
        """
        DDG Instant Answer API — returns JSON with AbstractURL and RelatedTopics.
        Very reliable, rarely blocked, but only gives 1-3 URLs for factual queries.
        """
        from urllib.parse import urlencode
        params = urlencode({
            "q": query, "format": "json",
            "no_html": "1", "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        resp = requests.get(url, headers=_make_headers(), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        urls = []
        # Main abstract URL
        if data.get("AbstractURL"):
            urls.append(data["AbstractURL"])
        # Official website
        if data.get("OfficialWebsite"):
            urls.append(data["OfficialWebsite"])
        # Related topics
        for topic in data.get("RelatedTopics", [])[:6]:
            if isinstance(topic, dict) and topic.get("FirstURL"):
                u = topic["FirstURL"]
                if u.startswith("http") and "duckduckgo.com" not in u:
                    urls.append(u)
        return urls

    # ── strategy 2: DuckDuckGo HTML with session + retry ─────────────────────

    def _search_ddg_html(self, query: str) -> List[str]:
        """
        DDG HTML search. Uses a requests.Session with cookies to appear
        more like a real browser, plus retries with different User-Agents.
        """
        from urllib.parse import unquote
        import time as _time

        for attempt in range(3):
            try:
                session = requests.Session()
                # First visit DDG homepage to get cookies
                session.get("https://duckduckgo.com/",
                            headers=_make_headers(), timeout=self.timeout)
                _time.sleep(0.5 + attempt * 0.5)  # polite delay

                url  = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}&kl=us-en"
                resp = session.get(url, headers=_make_headers(), timeout=self.timeout)
                resp.raise_for_status()

                soup = BeautifulSoup(resp.text, "html.parser")
                urls = []

                # Primary selector
                for a in soup.select("a.result__a"):
                    href = a.get("href", "")
                    clean = self._clean_ddg_href(href)
                    if clean: urls.append(clean)

                # Alternative selector (DDG sometimes changes class names)
                if not urls:
                    for a in soup.select("a[href*='uddg=']"):
                        href = a.get("href", "")
                        clean = self._clean_ddg_href(href)
                        if clean: urls.append(clean)

                if urls:
                    return urls
            except Exception as e:
                log.debug(f"DDG HTML attempt {attempt+1}: {e}")
        return []

    @staticmethod
    def _clean_ddg_href(href: str) -> Optional[str]:
        from urllib.parse import unquote
        if href.startswith("http") and "duckduckgo.com" not in href:
            return href
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            url = unquote(m.group(1))
            if url.startswith("http"): return url
        return None

    # ── strategy 3: SearXNG public instances ─────────────────────────────────

    def _search_searxng(self, query: str) -> List[str]:
        """
        SearXNG is a free open-source metasearch engine with public instances.
        Aggregates results from Google, Bing, DDG simultaneously.
        """
        from urllib.parse import urlencode
        for instance in self._SEARXNG_INSTANCES:
            try:
                params = urlencode({"q": query, "format": "json",
                                    "categories": "general", "lang": "en"})
                url  = f"{instance}?{params}"
                resp = requests.get(url, headers=_make_headers(),
                                    timeout=self.timeout)
                if resp.status_code != 200: continue
                data = resp.json()
                urls = [r["url"] for r in data.get("results", [])
                        if r.get("url","").startswith("http")]
                if urls:
                    return urls[:10]
            except Exception as e:
                log.debug(f"SearXNG {instance}: {e}")
        return []

    # ── strategy 4: Bing HTML ─────────────────────────────────────────────────

    def _search_bing(self, query: str) -> List[str]:
        """Bing HTML search with fresh session and realistic headers."""
        import time as _time
        for attempt in range(2):
            try:
                session = requests.Session()
                # Warm up session with Bing homepage
                session.get("https://www.bing.com/", headers=_make_headers(),
                            timeout=self.timeout)
                _time.sleep(0.3 + attempt * 0.5)

                url  = f"https://www.bing.com/search?q={quote_plus(query)}&setlang=en"
                resp = session.get(url, headers=_make_headers(),
                                   timeout=self.timeout)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                urls = []
                # Main result links
                for a in soup.select("li.b_algo h2 a, h2 a, .b_title a"):
                    href = a.get("href", "")
                    if href.startswith("http") and "bing.com" not in href:
                        urls.append(href)
                if urls: return list(dict.fromkeys(urls))  # dedup
            except Exception as e:
                log.debug(f"Bing attempt {attempt+1}: {e}")
        return []

    # ── strategy 5: Google (lite / simple endpoint) ───────────────────────────

    def _search_google(self, query: str) -> List[str]:
        """
        Uses Google's basic HTML search. Works when other engines are blocked.
        """
        try:
            url  = f"https://www.google.com/search?q={quote_plus(query)}&hl=en&num=10"
            # Google needs a very convincing UA
            ua   = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            hdrs = _make_headers(ua)
            hdrs["Referer"] = "https://www.google.com/"
            resp = requests.get(url, headers=hdrs, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            urls = []
            # Google wraps results in <a href="/url?q=ACTUAL_URL&...">
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/url?q="):
                    from urllib.parse import unquote
                    actual = unquote(href[7:].split("&")[0])
                    if actual.startswith("http") and "google.com" not in actual:
                        urls.append(actual)
            return list(dict.fromkeys(urls))[:10]
        except Exception as e:
            log.debug(f"Google error: {e}")
            return []

    # ── strategy 6: Wikipedia REST API ───────────────────────────────────────

    def _search_wikipedia_api(self, query: str) -> List[str]:
        """
        Wikipedia's public search API — always works, great for factual queries.
        Returns direct Wikipedia article URLs.
        """
        try:
            from urllib.parse import urlencode
            params = urlencode({"action":"opensearch","search":query,
                               "limit":"5","namespace":"0","format":"json"})
            url  = f"https://en.wikipedia.org/w/api.php?{params}"
            resp = requests.get(url, headers=_make_headers(), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # data[3] is the list of article URLs
            if len(data) >= 4:
                return [u for u in data[3] if u.startswith("http")]
        except Exception as e:
            log.debug(f"Wikipedia API error: {e}")
        return []

    def _wikipedia_direct(self, query: str) -> List[ScrapedPage]:
        """
        Emergency fallback: fetch Wikipedia article content directly via
        the Wikipedia API (returns plain text, no HTML scraping needed).
        """
        try:
            from urllib.parse import urlencode
            # Search for matching articles
            params = urlencode({"action":"query","list":"search",
                               "srsearch":query,"srlimit":"3",
                               "format":"json","utf8":"1"})
            url  = f"https://en.wikipedia.org/w/api.php?{params}"
            resp = requests.get(url, headers=_make_headers(), timeout=self.timeout)
            resp.raise_for_status()
            results = resp.json().get("query",{}).get("search",[])

            pages = []
            for result in results[:3]:
                title = result["title"]
                # Fetch article extract
                ep = urlencode({"action":"query","prop":"extracts",
                               "exintro":"1","explaintext":"1",
                               "titles":title,"format":"json"})
                er = requests.get(f"https://en.wikipedia.org/w/api.php?{ep}",
                                  headers=_make_headers(), timeout=self.timeout)
                er.raise_for_status()
                page_data = er.json().get("query",{}).get("pages",{})
                for pid, pdata in page_data.items():
                    extract = pdata.get("extract","")
                    if extract and len(extract.split()) > 50:
                        wiki_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ','_'))}"
                        pages.append(ScrapedPage(
                            url=wiki_url, title=title,
                            content=extract[:self.max_content_len * 5],
                            source="wikipedia.org",
                        ))
                        log.info(f"  ✅ Wikipedia: {title}")
            return pages
        except Exception as e:
            log.debug(f"Wikipedia direct fallback error: {e}")
            return []

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _skip(url: str) -> bool:
        try:
            return any(d in urlparse(url).netloc.lower() for d in _SKIP_DOMAINS)
        except Exception:
            return True

    def _fetch(self, url: str) -> Optional[ScrapedPage]:
        try:
            resp = requests.get(url, headers=_make_headers(),
                                timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
            if "text/html" not in resp.headers.get("content-type", ""):
                return None

            soup  = BeautifulSoup(resp.text, "html.parser")
            title = (soup.title.string or "").strip() if soup.title else ""

            # Strip noise tags
            for tag in soup(["script","style","nav","footer","header",
                              "aside","form","iframe","noscript"]):
                tag.decompose()
            for tag in soup.find_all(True):
                cls = " ".join(tag.get("class", []))
                tid = tag.get("id", "")
                if any(kw in cls.lower() or kw in tid.lower()
                       for kw in ["cookie","banner","popup","sidebar",
                                  "ad-","nav","menu","promo"]):
                    tag.decompose()

            # Prefer semantic content containers
            main = (soup.find("article") or
                    soup.find("main") or
                    soup.find(attrs={"role": "main"}) or
                    soup.find("div", id=re.compile(r"content|main|article", re.I)) or
                    soup.find("div", class_=re.compile(r"content|article|post|body|entry", re.I)) or
                    soup.body)
            if not main: return None

            paras = []
            for el in main.find_all(["p","h1","h2","h3","h4","li"]):
                t = el.get_text(" ", strip=True)
                if len(t.split()) > 5:
                    paras.append(t)

            content = re.sub(r"\n{3,}", "\n\n", "\n\n".join(paras)).strip()
            content = re.sub(r"[ \t]+", " ", content)
            if not content: return None

            words = content.split()
            if len(words) > self.max_content_len:
                content = " ".join(words[:self.max_content_len])

            domain = urlparse(url).netloc.replace("www.", "")
            return ScrapedPage(url=url, title=title or domain,
                               content=content, source=domain)
        except Exception as e:
            log.debug(f"Fetch error {url}: {e}")
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  §3  CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    text:     str
    source:   str
    url:      str
    chunk_id: str
    idx:      int
    tokens:   List[str] = field(default_factory=list)


class Chunker:
    _SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")
    _PARA = re.compile(r"\n{2,}")

    def __init__(self, size=300, overlap=50, min_size=30):
        self.size = size; self.overlap = overlap; self.min_size = min_size

    def chunk(self, text: str, source: str, url: str) -> List[Chunk]:
        paras = [p.strip() for p in self._PARA.split(text) if p.strip()]
        sents = []
        for p in paras:
            parts = [s.strip() for s in self._SENT.split(p) if s.strip()]
            sents.extend(parts if parts else [p])

        groups, buf, buf_wc = [], [], 0
        for sent in sents:
            wc = len(sent.split())
            if buf_wc + wc > self.size and buf:
                groups.append(" ".join(buf))
                ow, ob = 0, []
                for s in reversed(buf):
                    sw = len(s.split())
                    if ow + sw > self.overlap: break
                    ob.insert(0, s); ow += sw
                buf = ob; buf_wc = ow
            buf.append(sent); buf_wc += wc
        if buf: groups.append(" ".join(buf))

        chunks = []
        for i, t in enumerate(groups):
            if len(t.split()) < self.min_size: continue
            cid = hashlib.md5(f"{source}:{i}:{t[:64]}".encode()).hexdigest()[:16]
            chunks.append(Chunk(text=t, source=source, url=url,
                                chunk_id=cid, idx=i, tokens=tokenize(t)))
        return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  §4  BERT EMBEDDER
# ══════════════════════════════════════════════════════════════════════════════

class BERTEmbedder:
    _MODEL = "all-MiniLM-L6-v2"

    def __init__(self, dim=384):
        self.dim   = dim
        self._model = None
        self._mode  = "unloaded"
        self._proj  = None
        self._idf:  Dict[str, float] = {}

    def _load(self):
        if self._mode != "unloaded": return
        if _HAS_ST:
            try:
                log.info(f"Loading sentence-transformers ({self._MODEL}) …")
                self._model = SentenceTransformer(self._MODEL, device=_DEVICE)
                self.dim    = self._model.get_sentence_embedding_dimension()
                self._mode  = "st"
                log.info(f"✅ BERT  dim={self.dim}  device={_DEVICE}")
                return
            except Exception as e:
                log.warning(f"sentence-transformers failed: {e}")
        log.info("Using hash embedder fallback")
        rng = np.random.default_rng(42)
        nb  = 2**17
        self._proj = rng.standard_normal((nb, self.dim)).astype(np.float32)
        n = np.linalg.norm(self._proj, axis=1, keepdims=True)
        n[n < 1e-9] = 1.0
        self._proj /= n
        self._mode = "hash"

    def fit_idf(self, token_lists):
        self._load()
        if self._mode == "hash":
            N = len(token_lists) or 1
            df: Counter = Counter()
            for tl in token_lists: df.update(set(tl))
            self._idf = {t: math.log(1 + N/(c+1)) for t, c in df.items()}

    def embed(self, text_or_tokens) -> np.ndarray:
        self._load()
        text = " ".join(text_or_tokens) if isinstance(text_or_tokens, list) else str(text_or_tokens)
        if self._mode == "st":
            v = self._model.encode(text, convert_to_numpy=True,
                                   show_progress_bar=False, normalize_embeddings=True)
            return v.astype(np.float32)
        return self._hash_embed(tokenize(text))

    def _hash_embed(self, tokens):
        tf = Counter(tokens); n = len(tokens) or 1
        vec = np.zeros(2**17, dtype=np.float32)
        for tok, cnt in tf.items():
            w   = (cnt/n) * self._idf.get(tok, 1.0)
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % (2**17)
            vec[idx] += w
        out  = vec @ self._proj
        norm = np.linalg.norm(out)
        return out / norm if norm > 1e-9 else out

    def embed_batch(self, texts, batch_size=64):
        self._load()
        if self._mode == "st":
            v = self._model.encode(texts, batch_size=batch_size,
                                   convert_to_numpy=True, show_progress_bar=False,
                                   normalize_embeddings=True)
            return v.astype(np.float32)
        return np.stack([self.embed(t) for t in texts])


# ══════════════════════════════════════════════════════════════════════════════
#  §5  BM25+ INDEX
# ══════════════════════════════════════════════════════════════════════════════

class BM25Index:
    def __init__(self, k1=1.2, b=0.75, delta=1.0):
        self.k1=k1; self.b=b; self.delta=delta
        self._ids=[]; self._toks=[]; self._idf={}
        self._tf_docs=[]; self._avgdl=1.0; self._built=False

    def add(self, cid, tokens):
        self._ids.append(cid); self._toks.append(tokens); self._built=False

    def build(self):
        N = len(self._toks)
        if N == 0: return
        dl = [len(t) for t in self._toks]
        self._avgdl = sum(dl)/N or 1.0
        df: Counter = Counter()
        for t in self._toks: df.update(set(t))
        self._idf = {term: math.log(1+(N-f+0.5)/(f+0.5)) for term,f in df.items()}
        self._tf_docs = [Counter(t) for t in self._toks]
        self._built = True

    def search(self, qtoks, k):
        if not self._built: self.build()
        if not self._ids: return []
        scores = np.zeros(len(self._ids), dtype=np.float64)
        for term in set(qtoks):
            if term not in self._idf: continue
            idf = self._idf[term]
            for i, (tf_d, toks) in enumerate(zip(self._tf_docs, self._toks)):
                tf = tf_d.get(term, 0)
                if tf == 0: continue
                tfn = (tf*(self.k1+1))/(tf+self.k1*(1-self.b+self.b*len(toks)/self._avgdl))
                scores[i] += idf*(self.delta+tfn)
        k = min(k, len(self._ids))
        if k == 0: return []
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [(self._ids[i], float(scores[i])) for i in top if scores[i] > 0]

    def clear(self):
        self._ids.clear(); self._toks.clear()
        self._idf.clear(); self._tf_docs.clear(); self._built=False

    def __len__(self): return len(self._ids)


# ══════════════════════════════════════════════════════════════════════════════
#  §6  CHROMADB VECTOR STORE  (with numpy fallback)
# ══════════════════════════════════════════════════════════════════════════════

class VectorStore:
    """
    ChromaDB persistent vector store.
    Falls back to numpy in-memory if chromadb not installed.
    """
    COLLECTION = "rag_web_v1"

    def __init__(self, persist_dir=".rag_chroma_db", dim=384):
        self.dim  = dim
        self._mode = "none"
        self._col  = None
        self._cli  = None
        # numpy fallback
        self._np_ids: List[str] = []
        self._np_mat: Optional[np.ndarray] = None

        if _HAS_CHROMA:
            try:
                self._cli = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )
                self._col = self._cli.get_or_create_collection(
                    name=self.COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
                self._mode = "chroma"
                log.info(f"✅ ChromaDB  path={persist_dir}  docs={self._col.count()}")
            except Exception as e:
                log.warning(f"ChromaDB init failed ({e}) — using numpy")
                self._mode = "numpy"
        else:
            self._mode = "numpy"

    def add_batch(self, ids, embeddings, metadatas, documents):
        if self._mode == "chroma":
            self._col.upsert(ids=ids, embeddings=embeddings.tolist(),
                             metadatas=metadatas, documents=documents)
        else:
            for cid, emb in zip(ids, embeddings):
                v = emb.astype(np.float32).reshape(1, -1)
                self._np_mat = v if self._np_mat is None else np.vstack([self._np_mat, v])
                if cid not in self._np_ids: self._np_ids.append(cid)

    def search(self, qvec, k) -> List[Tuple[str, float]]:
        if self._mode == "chroma":
            cnt = self._col.count()
            if cnt == 0: return []
            k = min(k, cnt)
            res  = self._col.query(query_embeddings=[qvec.tolist()],
                                   n_results=k, include=["distances"])
            return [(cid, 1.0 - dist)
                    for cid, dist in zip(res["ids"][0], res["distances"][0])]
        else:
            if self._np_mat is None or not self._np_ids: return []
            scores = self._np_mat @ qvec.astype(np.float32)
            k = min(k, len(self._np_ids))
            top = np.argpartition(scores, -k)[-k:]
            top = top[np.argsort(scores[top])[::-1]]
            return [(self._np_ids[i], float(scores[i])) for i in top]

    def clear(self):
        if self._mode == "chroma" and self._cli:
            try:
                self._cli.delete_collection(self.COLLECTION)
                self._col = self._cli.get_or_create_collection(
                    name=self.COLLECTION,
                    metadata={"hnsw:space": "cosine"},
                )
                log.info("ChromaDB cleared.")
            except Exception as e:
                log.warning(f"ChromaDB clear error: {e}")
        else:
            self._np_ids.clear(); self._np_mat = None

    def count(self):
        return self._col.count() if self._mode == "chroma" else len(self._np_ids)

    @property
    def mode(self): return self._mode


# ══════════════════════════════════════════════════════════════════════════════
#  §7  CRAG EVALUATOR  (corrective retrieval quality gating)
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalQuality(Enum):
    CORRECT   = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class Hit:
    chunk:        Chunk
    bm25_score:   float = 0.0
    vec_score:    float = 0.0
    hybrid_score: float = 0.0
    crag_score:   float = 0.0
    crag_quality: RetrievalQuality = RetrievalQuality.AMBIGUOUS
    embedding:    Optional[np.ndarray] = field(default=None, repr=False)


class CRAGEvaluator:
    CORRECT_T   = 0.55
    INCORRECT_T = 0.25

    def __init__(self, bm25_w=0.4, vec_w=0.6):
        self.bm25_w = bm25_w; self.vec_w = vec_w

    def evaluate(self, qtoks, qvec, hits):
        q_set = set(qtoks)
        for h in hits:
            c_set   = set(h.chunk.tokens)
            bm25_ov = len(q_set & c_set) / len(q_set) if q_set else 0.0
            if h.embedding is not None and qvec is not None:
                cos = max(0.0, float(np.dot(h.embedding, qvec) /
                                     (np.linalg.norm(h.embedding) *
                                      np.linalg.norm(qvec) + 1e-9)))
            else:
                cos = h.vec_score
            h.crag_score = self.bm25_w * bm25_ov + self.vec_w * cos
            if h.crag_score >= self.CORRECT_T:
                h.crag_quality = RetrievalQuality.CORRECT
            elif h.crag_score >= self.INCORRECT_T:
                h.crag_quality = RetrievalQuality.AMBIGUOUS
            else:
                h.crag_quality = RetrievalQuality.INCORRECT
        return hits

    @staticmethod
    def refine_ambiguous(hit, qtoks):
        sents  = re.split(r"(?<=[.!?])\s+", hit.chunk.text)
        q_set  = set(qtoks)
        scored = sorted(
            [(len(q_set & set(tokenize(s))) / max(len(q_set), 1), s) for s in sents],
            key=lambda x: -x[0]
        )
        keep = max(1, len(scored) // 2)
        new_text = " ".join(s for _, s in scored[:keep])
        hit.chunk = Chunk(text=new_text, source=hit.chunk.source,
                          url=hit.chunk.url, chunk_id=hit.chunk.chunk_id+"_r",
                          idx=hit.chunk.idx, tokens=tokenize(new_text))
        return hit


# ══════════════════════════════════════════════════════════════════════════════
#  §8  MMR RERANKER
# ══════════════════════════════════════════════════════════════════════════════

def mmr_rerank(hits, qvec, k, lam=0.65):
    if len(hits) <= k: return hits
    selected, pool = [], list(hits)
    while len(selected) < k and pool:
        if not selected: selected.append(pool.pop(0)); continue
        sel = np.stack([h.embedding for h in selected if h.embedding is not None])
        best_score, best_i = -1e9, 0
        for i, h in enumerate(pool):
            rel = h.hybrid_score
            div = float((sel @ h.embedding).max()) if h.embedding is not None else 0.0
            score = lam * rel - (1-lam) * div
            if score > best_score: best_score = score; best_i = i
        selected.append(pool.pop(best_i))
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  §9  TEXTRANK
# ══════════════════════════════════════════════════════════════════════════════

class TextRank:
    def __init__(self, d=0.85, iters=30):
        self.d = d; self.iters = iters

    def summarise(self, text, n=4):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                 if len(s.split()) > 4]
        if len(sents) <= n: return text
        tl    = [tokenize(s) for s in sents]
        vocab = sorted({t for x in tl for t in x})
        v2i   = {t: i for i, t in enumerate(vocab)}
        V, N  = len(vocab), len(sents)
        mat   = np.zeros((N, V), dtype=np.float32)
        for i, t in enumerate(tl):
            c = Counter(t); total = len(t) or 1
            for tok, cnt in c.items():
                if tok in v2i: mat[i, v2i[tok]] = cnt/total
        df = (mat > 0).sum(axis=0) + 1
        idf = np.log((N+1)/df).astype(np.float32)
        mat  = mat * idf
        nms  = np.linalg.norm(mat, axis=1, keepdims=True)
        nms[nms < 1e-9] = 1.0
        mat /= nms
        sim  = mat @ mat.T
        np.fill_diagonal(sim, 0.0)
        rs   = sim.sum(axis=1, keepdims=True)
        rs[rs < 1e-9] = 1.0
        trans = sim / rs
        scores = np.ones(N, dtype=np.float32) / N
        for _ in range(self.iters):
            scores = (1-self.d)/N + self.d*(trans.T @ scores)
        top = sorted(np.argsort(scores)[-n:])
        return " ".join(sents[i] for i in top)


# ══════════════════════════════════════════════════════════════════════════════
#  §10  DEEPSEEK / OLLAMA LLM
# ══════════════════════════════════════════════════════════════════════════════

class DeepSeekLLM:
    """
    Calls DeepSeek-R1 (or any model) via Ollama.
    Falls back to FLAN-T5 → GPT-2 → TextRank.

    Setup:
      1. Install Ollama: https://ollama.com
      2. ollama pull deepseek-r1:7b
      3. ollama serve
    """

    OLLAMA_URL = "http://localhost:11434/api/generate"

    def __init__(self, model="deepseek-r1:7b", max_tokens=400, timeout=60):
        self.model      = model
        self.max_tokens = max_tokens
        self.timeout    = timeout
        self._mode      = "undetected"
        self._hf_model  = None
        self._hf_tok    = None
        self._hf_type   = None
        self._tr        = TextRank()
        self._detect()

    def _detect(self):
        import urllib.request as urlreq
        # 1. Ollama
        try:
            r    = urlreq.urlopen("http://localhost:11434/api/tags", timeout=3)
            data = json.loads(r.read())
            avail = [m["name"] for m in data.get("models", [])]
            log.info(f"Ollama running. Models: {avail}")
            for m in [self.model, "llama3.2", "llama3", "mistral"] + avail:
                base = m.split(":")[0]
                if any(base in a for a in avail):
                    self.model = m; self._mode = "ollama"
                    log.info(f"✅ Ollama model: {self.model}")
                    return
            log.warning("Ollama running but no models. Run: ollama pull deepseek-r1:7b")
        except Exception:
            log.info("Ollama not running — trying transformers …")

        # 2. FLAN-T5
        if _HAS_TRANSFORMERS and _HAS_TORCH:
            try:
                log.info("Loading FLAN-T5-small …")
                self._hf_tok   = T5Tokenizer.from_pretrained("google/flan-t5-small", legacy=False)
                self._hf_model = T5ForConditionalGeneration.from_pretrained(
                    "google/flan-t5-small").to(_DEVICE)
                self._hf_model.eval()
                self._hf_type  = "t5"
                self._mode     = "huggingface"
                log.info(f"✅ FLAN-T5-small  device={_DEVICE}")
                return
            except Exception as e:
                log.warning(f"FLAN-T5 failed: {e}")

            # 3. GPT-2
            try:
                log.info("Loading GPT-2 …")
                self._hf_tok            = GPT2Tokenizer.from_pretrained("gpt2")
                self._hf_tok.pad_token  = self._hf_tok.eos_token
                self._hf_model          = GPT2LMHeadModel.from_pretrained("gpt2").to(_DEVICE)
                self._hf_model.eval()
                self._hf_type           = "gpt2"
                self._mode              = "huggingface"
                log.info(f"✅ GPT-2  device={_DEVICE}")
                return
            except Exception as e:
                log.warning(f"GPT-2 failed: {e}")

        # 4. TextRank
        log.info("Using TextRank extractive fallback.")
        self._mode = "textrank"

    def generate(self, query, context):
        if self._mode   == "ollama":       return self._ollama(query, context)
        elif self._mode == "huggingface":  return self._hf(query, context)
        else:                              return self._tr.summarise(context, n=4)

    def _ollama(self, query, context):
        import urllib.request as urlreq
        ctx = " ".join(context.split()[:1200])
        prompt = (
            f"You are a helpful assistant. Answer the question using ONLY the "
            f"context below. Be clear and concise.\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        payload = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"num_predict": self.max_tokens, "temperature": 0.1},
        }).encode()
        try:
            req = urlreq.Request(self.OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
            with urlreq.urlopen(req, timeout=self.timeout) as r:
                ans = json.loads(r.read()).get("response", "").strip()
                # Strip DeepSeek <think>...</think> blocks
                ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()
                return ans or "(no answer generated)"
        except Exception as e:
            log.warning(f"Ollama error: {e}")
            return self._tr.summarise(context, n=4)

    def _hf(self, query, context):
        ctx    = " ".join(context.split()[:350])
        prompt = f"Answer based on context.\nContext: {ctx}\nQuestion: {query}\nAnswer:"
        with torch.no_grad():
            enc = self._hf_tok(prompt, return_tensors="pt",
                               max_length=512, truncation=True).to(_DEVICE)
            if self._hf_type == "t5":
                out = self._hf_model.generate(**enc, max_new_tokens=150,
                                              num_beams=4, early_stopping=True)
                ans = self._hf_tok.decode(out[0], skip_special_tokens=True)
            else:
                il  = enc["input_ids"].shape[1]
                out = self._hf_model.generate(**enc, max_new_tokens=100,
                                              do_sample=False,
                                              pad_token_id=self._hf_tok.eos_token_id)
                ans = self._hf_tok.decode(out[0][il:], skip_special_tokens=True)
                for end in ["\n","? ","! ",". "]:
                    idx = ans.find(end)
                    if idx != -1: ans = ans[:idx+1]; break
        return ans.strip() or "(no answer)"

    @property
    def mode(self): return self._mode


# ══════════════════════════════════════════════════════════════════════════════
#  §11  MAIN RAG-WEB ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RAGWebConfig:
    max_pages:       int   = 5
    scrape_timeout:  int   = 10
    max_content_len: int   = 6000
    chunk_size:      int   = 300
    chunk_overlap:   int   = 50
    min_chunk_size:  int   = 30
    embed_dim:       int   = 384
    bm25_k:          int   = 20
    vec_k:           int   = 20
    final_k:         int   = 5
    mmr_lambda:      float = 0.65
    bm25_w:          float = 0.35
    vec_w:           float = 0.65
    use_crag:        bool  = True
    ollama_model:    str   = "deepseek-r1:7b"
    chroma_dir:      str   = ".rag_chroma_db"
    cache_scrapes:   bool  = True


class RAGWeb:
    """
    Full pipeline:
    Query → Web scrape → Chunk → BERT embed → ChromaDB store
         → BM25 + vector hybrid → CRAG gate → MMR → DeepSeek answer
    """

    def __init__(self, cfg: Optional[RAGWebConfig] = None):
        self.cfg      = cfg or RAGWebConfig()
        self._scraper  = WebScraper(self.cfg.max_pages,
                                    self.cfg.scrape_timeout,
                                    self.cfg.max_content_len)
        self._chunker  = Chunker(self.cfg.chunk_size,
                                 self.cfg.chunk_overlap,
                                 self.cfg.min_chunk_size)
        self._embedder = BERTEmbedder(self.cfg.embed_dim)
        self._vec      = VectorStore(self.cfg.chroma_dir, self.cfg.embed_dim)
        self._bm25     = BM25Index()
        self._crag     = CRAGEvaluator()
        self._llm      = DeepSeekLLM(self.cfg.ollama_model)
        self._tr       = TextRank()
        self._chunks:     Dict[str, Chunk]       = {}
        self._embeddings: Dict[str, np.ndarray]  = {}
        self._cache:      Dict[str, List[ScrapedPage]] = {}

    def query(self, question: str) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # ── Step 1: Scrape ────────────────────────────────────────────────────
        pages = self._get_pages(question)
        if not pages:
            return {
                "question":   question,
                "llm_answer": "❌ Could not retrieve web pages. Check your internet connection.",
                "extractive": "",
                "hits": [], "sources": [],
                "pages_scraped": 0, "chunks_indexed": 0,
                "latency_ms": round((time.perf_counter()-t0)*1000, 2),
                "llm_mode": self._llm.mode,
                "vec_mode": self._vec.mode,
            }

        # ── Step 2: Chunk ─────────────────────────────────────────────────────
        all_chunks: List[Chunk] = []
        for page in pages:
            chunks = self._chunker.chunk(page.content, page.source, page.url)
            all_chunks.extend(chunks)

        log.info(f"📦 {len(all_chunks)} chunks from {len(pages)} pages")

        if not all_chunks:
            log.warning("Pages were fetched but chunking produced 0 chunks (content too short?)")
            # Try re-chunking with lower min_size as fallback
            tmp_chunker = Chunker(size=self.cfg.chunk_size,
                                   overlap=self.cfg.chunk_overlap,
                                   min_size=10)
            for page in pages:
                all_chunks.extend(tmp_chunker.chunk(page.content, page.source, page.url))
            log.info(f"📦 After lowered min_size: {len(all_chunks)} chunks")

        # ── Step 3: Embed + store in ChromaDB ────────────────────────────────
        new = [c for c in all_chunks if c.chunk_id not in self._chunks]
        if new:
            texts = [c.text for c in new]
            log.info(f"🔢 Embedding {len(new)} new chunks …")
            vecs  = self._embedder.embed_batch(texts, batch_size=32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            vecs /= norms

            self._vec.add_batch(
                ids=[c.chunk_id for c in new],
                embeddings=vecs,
                metadatas=[{"source": c.source, "url": c.url} for c in new],
                documents=[c.text for c in new],
            )
            for chunk, vec in zip(new, vecs):
                self._chunks[chunk.chunk_id]     = chunk
                self._embeddings[chunk.chunk_id] = vec
                self._bm25.add(chunk.chunk_id, chunk.tokens)

            self._bm25.build()
            self._embedder.fit_idf([c.tokens for c in self._chunks.values()])
            log.info(f"✅ Index now has {len(self._chunks)} total chunks")

        # ── Step 4: Embed query ───────────────────────────────────────────────
        qvec  = self._embedder.embed(question)
        n     = np.linalg.norm(qvec)
        if n > 1e-9: qvec /= n
        qtoks = tokenize(question)

        # ── Step 5: Hybrid retrieval (BM25 + vector) ──────────────────────────
        bm25_res = dict(self._bm25.search(qtoks, self.cfg.bm25_k))
        vec_res  = dict(self._vec.search(qvec, self.cfg.vec_k))

        log.info(f"🔍 BM25 hits: {len(bm25_res)}  Vector hits: {len(vec_res)}")

        def norm_d(d: dict) -> dict:
            if not d: return d
            mx = max(d.values()) or 1.0
            return {k: v/mx for k, v in d.items()}

        bn = norm_d(bm25_res)
        vn = norm_d(vec_res)

        hits: List[Hit] = []
        for cid in set(bn) | set(vn):
            if cid not in self._chunks: continue
            b = bn.get(cid, 0.0)
            v = vn.get(cid, 0.0)
            hits.append(Hit(
                chunk        = self._chunks[cid],
                bm25_score   = b,
                vec_score    = v,
                hybrid_score = self.cfg.bm25_w * b + self.cfg.vec_w * v,
                embedding    = self._embeddings.get(cid),
            ))

        hits.sort(key=lambda x: -x.hybrid_score)
        log.info(f"📊 Hybrid hits before CRAG: {len(hits)}")

        # ── Step 6: CRAG quality gating ───────────────────────────────────────
        # NOTE: CRAG is used to IMPROVE hits, not drop them aggressively.
        # We always keep at least final_k chunks regardless of CRAG score.
        if self.cfg.use_crag and hits:
            hits = self._crag.evaluate(qtoks, qvec, hits)

            correct   = [h for h in hits if h.crag_quality == RetrievalQuality.CORRECT]
            ambiguous = [h for h in hits if h.crag_quality == RetrievalQuality.AMBIGUOUS]
            incorrect = [h for h in hits if h.crag_quality == RetrievalQuality.INCORRECT]

            log.info(f"🔬 CRAG: {len(correct)} correct  "
                     f"{len(ambiguous)} ambiguous  {len(incorrect)} incorrect")

            refined = []
            # Keep all correct chunks as-is
            refined.extend(correct)
            # Refine ambiguous (keep best sentences)
            for h in ambiguous:
                refined.append(self._crag.refine_ambiguous(h, qtoks))

            # IMPORTANT FIX: if we don't have enough after filtering,
            # fall back to top hybrid-scored chunks (don't silently return 0)
            if len(refined) < self.cfg.final_k:
                needed = self.cfg.final_k - len(refined)
                # Add best incorrect chunks as last resort
                already = {h.chunk.chunk_id for h in refined}
                for h in incorrect:
                    if h.chunk.chunk_id not in already:
                        refined.append(h)
                        needed -= 1
                        if needed <= 0: break

            hits = refined

        # ── Step 7: MMR reranking ─────────────────────────────────────────────
        hits = mmr_rerank(hits, qvec, self.cfg.final_k, self.cfg.mmr_lambda)
        log.info(f"✅ Final hits after MMR: {len(hits)}")

        # ── Step 8: Generate answer ───────────────────────────────────────────
        if not hits:
            log.warning("0 hits after all filtering — returning raw page snippet")
            # Emergency: just use first scraped page content directly
            raw_ctx = pages[0].content[:1500] if pages else ""
            answer  = self._llm.generate(question, raw_ctx)
            extract = self._tr.summarise(raw_ctx, n=3) if raw_ctx else ""
        else:
            context = " ".join(h.chunk.text for h in hits)
            answer  = self._llm.generate(question, context)
            extract = self._tr.summarise(context, n=3)

        ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "question":       question,
            "llm_answer":     answer,
            "extractive":     extract,
            "hits":           hits,
            "sources":        list({h.chunk.url for h in hits}),
            "pages_scraped":  len(pages),
            "chunks_indexed": len(self._chunks),
            "latency_ms":     ms,
            "llm_mode":       self._llm.mode,
            "vec_mode":       self._vec.mode,
        }

    def _get_pages(self, query: str):
        # Normalise cache key: lowercase, strip punctuation
        import re as _re
        cache_key = _re.sub(r"\W+", " ", query.lower()).strip()
        if self.cfg.cache_scrapes and cache_key in self._cache:
            log.info(f"Using cached scrape ({len(self._cache[cache_key])} pages)")
            return self._cache[cache_key]
        pages = self._scraper.scrape_query(query)
        if self.cfg.cache_scrapes:
            self._cache[cache_key] = pages
        return pages

    def clear(self):
        self._vec.clear(); self._bm25.clear()
        self._chunks.clear(); self._embeddings.clear(); self._cache.clear()
        log.info("✅ Cleared.")

    def stats(self):
        return {
            "chunks_in_memory": len(self._chunks),
            "chroma_docs":      self._vec.count(),
            "vec_mode":         self._vec.mode,
            "llm_mode":         self._llm.mode,
            "model":            self._llm.model,
            "bert_mode":        self._embedder._mode,
            "cached_queries":   list(self._cache.keys()),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  §12  CLI
# ══════════════════════════════════════════════════════════════════════════════

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║  SuperiorRAG  WEB EDITION                                           ║
║  Web Scraper → ChromaDB → BERT → BM25+Vector → CRAG → DeepSeek    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

SEP = "─" * 70


def _print_result(r: Dict[str, Any]):
    print(f"\n{SEP}")
    print(f"❓  {r['question']}")
    print(SEP)

    mode_label = {
        "ollama":      "🤖 DeepSeek / Ollama",
        "huggingface": "🤖 FLAN-T5 / GPT-2",
        "textrank":    "📄 Extractive",
    }.get(r.get("llm_mode",""), "🤖 LLM")

    if r.get("llm_answer"):
        print(f"\n{mode_label}:")
        for line in textwrap.wrap(r["llm_answer"], width=68):
            print(f"  {line}")

    if r.get("extractive") and r.get("llm_mode") != "textrank":
        print(f"\n📄 Extractive (TextRank):")
        for line in textwrap.wrap(r["extractive"][:400], width=68):
            print(f"  {line}")

    hits = r.get("hits", [])
    if hits:
        q = Counter({"✅":0,"🟡":0,"❌":0})
        q.update({"✅" if h.crag_quality==RetrievalQuality.CORRECT
                  else "🟡" if h.crag_quality==RetrievalQuality.AMBIGUOUS
                  else "❌": 1 for h in hits})
        print(f"\n🔍 CRAG: {q['✅']} correct  {q['🟡']} ambiguous  {q['❌']} incorrect")

    if r.get("sources"):
        print(f"\n🌐 Sources ({r.get('pages_scraped',0)} pages scraped):")
        for url in list(r["sources"])[:5]:
            print(f"   • {url}")

    print(f"\n⏱  {r['latency_ms']} ms  |  {len(hits)} chunks  |  "
          f"DB: {r.get('vec_mode','?')}  |  LLM: {r.get('llm_mode','?')}")
    print(SEP)


def _print_stats(rag: RAGWeb):
    s = rag.stats()
    print(f"\n{'═'*50}")
    print("  SYSTEM STATUS")
    print(f"{'═'*50}")
    print(f"  Chunks (memory) : {s['chunks_in_memory']}")
    print(f"  ChromaDB docs   : {s['chroma_docs']}")
    print(f"  Vector store    : {s['vec_mode']}")
    print(f"  LLM backend     : {s['llm_mode']}")
    print(f"  Model           : {s['model']}")
    print(f"  BERT mode       : {s['bert_mode']}")
    if s["cached_queries"]:
        print(f"  Cached queries  : {s['cached_queries']}")
    print(f"{'═'*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="SuperiorRAG Web — live scraping + DeepSeek + ChromaDB"
    )
    parser.add_argument("--query",   type=str,  help="Ask a single question")
    parser.add_argument("--sources", type=int,  default=5,
                        help="Pages to scrape per query (default: 5)")
    parser.add_argument("--model",   type=str,  default="deepseek-r1:7b",
                        help="Ollama model (default: deepseek-r1:7b)")
    parser.add_argument("--clear",   action="store_true", help="Clear ChromaDB")
    parser.add_argument("--stats",   action="store_true", help="Show system stats")
    parser.add_argument("--no-crag", action="store_true", help="Disable CRAG")
    parser.add_argument("--debug",   action="store_true", help="Show verbose debug logs")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.setLevel(logging.DEBUG)

    print(BANNER)

    cfg = RAGWebConfig(
        max_pages    = args.sources,
        ollama_model = args.model,
        use_crag     = not args.no_crag,
    )
    rag = RAGWeb(cfg)

    if args.clear:
        rag.clear(); print("✅ Cleared."); return
    if args.stats:
        _print_stats(rag); return
    if args.query:
        _print_result(rag.query(args.query)); return

    # Interactive chat
    print("💬  Ask anything — I'll scrape the web in real time and answer.")
    print("    Commands: 'stats'  'clear'  'exit'\n")

    while True:
        try:
            q = input("You › ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye! 👋"); break
        if not q: continue
        if q.lower() in ("exit","quit","q"): print("Bye! 👋"); break
        if q.lower() == "stats":  _print_stats(rag); continue
        if q.lower() == "clear":  rag.clear(); print("✅ Cleared."); continue
        _print_result(rag.query(q))


if __name__ == "__main__":
    main()
