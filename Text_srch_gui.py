"""
Text Search Engine
==================
Pipeline:  .txt file -> document splitting -> tokenization -> linguistic
           processing (lowercase, stop-words, stemming) -> positional
           inverted index (+ k-gram index + Soundex index) -> query engine -> GUI

Query types (Box 1):
    information retrieval          ranked free-text (implicit AND)
    cat AND dog / cat OR dog       boolean
    cat AND NOT dog                negation
    (cat OR dog) AND food          parentheses
    "machine learning"             phrase (uses positional index)
    comp*  *tion  c?t  a*b*c       wildcard (k-gram index + regex filter)
    soundex:smith  or  ~smith      phonetic match (Soundex)
    informaton                     misspelled word -> auto spell-correct
Run:  python text_search_engine.py
"""
import re
import math
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from collections import defaultdict, Counter

# --------------------------------------------------------------------------
# 1. TOKENIZATION
# --------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def tokenize(text):
    """Yield (lowercase_token, position, start_char, end_char)."""
    for pos, m in enumerate(TOKEN_RE.finditer(text)):
        yield m.group().lower(), pos, m.start(), m.end()


# --------------------------------------------------------------------------
# 2. LINGUISTIC MODELS: stop-words, stemming, Soundex, edit distance
# --------------------------------------------------------------------------
STOP = set("""a an the and or not of to in on at by for with from as is are was were
be been being it its this that these those i you he she we they them his her their
our your my me do does did has have had but if then so than too very can will just
into over under about up out no nor""".split())

try:  # use the real Porter stemmer when NLTK is available
    from nltk.stem import PorterStemmer
    _porter = PorterStemmer()

    def stem(w):
        return _porter.stem(w)
except Exception:  # lightweight fallback stemmer
    def stem(w):
        if len(w) <= 3 or w.endswith("ss"):
            return w
        for suf in ("ingly", "edly", "ing", "ed", "ly", "ies", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                return w + "y" if suf == "ies" else w
        return w

_SDX = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}


def soundex(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return ""
    out, last = word[0].upper(), _SDX.get(word[0], "")
    for ch in word[1:]:
        c = _SDX.get(ch, "")
        if c and c != last:
            out += c
        if ch not in "hw":
            last = c
    return (out + "000")[:4]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def kgrams(word, k=2):
    w = f"${word}$"
    return {w[i:i + k] for i in range(len(w) - k + 1)}


# --------------------------------------------------------------------------
# 3. INDEXING + 4. QUERY ENGINE
# --------------------------------------------------------------------------
class SearchEngine:
    def __init__(self):
        self.docs = []
        self.index = {}       # stem -> {doc_id: [positions]}   (positional inverted index)
        self.raw_vocab = {}   # surface word -> stem
        self.kgram = {}       # bigram -> set(surface words)    (wildcard + spelling)
        self.sdx = {}         # soundex code -> set(surface words)

    # ---------- build ----------
    def build(self, text):
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paras) <= 1:  # no blank-line paragraphs -> one document per line
            paras = [l.strip() for l in text.splitlines() if l.strip()]
        self.docs = paras
        index = defaultdict(lambda: defaultdict(list))
        raw = {}
        for d, doc in enumerate(self.docs):
            for tok, pos, _, _ in tokenize(doc):
                if tok in STOP:
                    continue  # positions keep counting, so phrase gaps stay correct
                s = stem(tok)
                index[s][d].append(pos)
                raw[tok] = s
        self.index = {s: dict(p) for s, p in index.items()}
        self.raw_vocab = raw
        self.kgram, self.sdx = defaultdict(set), defaultdict(set)
        for w in raw:
            for g in kgrams(w):
                self.kgram[g].add(w)
            self.sdx[soundex(w)].add(w)
        return len(self.docs), len(self.raw_vocab)

    # ---------- helpers ----------
    def _docs_for_stem(self, s):
        return set(self.index.get(s, {}))

    def _docs_for_raw(self, terms):
        out = set()
        for t in terms:
            s = self.raw_vocab[t]
            self.qstems.add(s)
            out |= self._docs_for_stem(s)
        return out

    def wildcard_terms(self, pat):
        padded = f"${pat}$"
        cand = None
        for piece in re.split(r"[*?]", padded):
            for g in ({piece[j:j + 2] for j in range(len(piece) - 1)} if len(piece) >= 2 else ()):
                s = self.kgram.get(g, set())
                cand = set(s) if cand is None else cand & s
        if cand is None:
            cand = set(self.raw_vocab)
        rx = re.compile("".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pat))
        return sorted(w for w in cand if rx.fullmatch(w))

    def correct(self, word):
        grams = kgrams(word)
        cnt = Counter()
        for g in grams:
            for w in self.kgram.get(g, ()):
                cnt[w] += 1
        best = None
        for w, c in cnt.items():
            if c / (len(grams) + len(kgrams(w)) - c) < 0.2:
                continue
            d = edit_distance(word, w)
            if d > max(1, len(word) // 3):
                continue
            key = (d, -len(self.index[self.raw_vocab[w]]), w)
            if best is None or key < best:
                best = key
        return best[2] if best else None

    # ---------- term / phrase evaluation ----------
    def term(self, t):
        low = t.lower()
        if low.startswith("soundex:") or low.startswith("~"):
            w = low.split(":", 1)[1] if ":" in low else low[1:]
            code = soundex(w)
            terms = sorted(self.sdx.get(code, ()))
            self.notes.append(f"Soundex {code} for '{w}': {', '.join(terms) or 'no matches'}")
            return self._docs_for_raw(terms)
        if "*" in low or "?" in low:
            terms = self.wildcard_terms(low)
            self.notes.append(f"Wildcard '{low}': {', '.join(terms[:15]) or 'no matches'}"
                              + (" ..." if len(terms) > 15 else ""))
            return self._docs_for_raw(terms)
        w = re.sub(r"[^a-z0-9']", "", low)
        if not w:
            return None
        if w in STOP:
            return None  # stop-word: ignored
        s = stem(w)
        if s in self.index:
            self.qstems.add(s)
            return self._docs_for_stem(s)
        c = self.correct(w)
        if c:
            self.notes.append(f"Spell-corrected '{w}' -> '{c}'")
            s = self.raw_vocab[c]
            self.qstems.add(s)
            return self._docs_for_stem(s)
        self.notes.append(f"No match for '{w}'")
        return set()

    def phrase(self, text):
        keep = [(stem(t), p) for t, p, _, _ in tokenize(text) if t not in STOP]
        if not keep:
            return None
        for s, _ in keep:
            if s not in self.index:
                self.notes.append(f"Phrase word '{s}' not in index")
                return set()
            self.qstems.add(s)
        base = keep[0][1]
        res = set()
        for d, plist in self.index[keep[0][0]].items():
            sets = []
            for s, _ in keep:
                if d not in self.index[s]:
                    break
                sets.append(set(self.index[s][d]))
            else:
                if any(all((p0 + off - base) in ps for (_, off), ps in zip(keep, sets)) for p0 in plist):
                    res.add(d)
        return res

    # ---------- boolean parser ----------
    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def p_or(self):
        left = self.p_and()
        while self.peek() == "OR":
            self.i += 1
            right = self.p_and()
            left = right if left is None else left if right is None else left | right
        return left

    def p_and(self):
        left = self.p_not()
        while self.peek() not in (None, "OR", ")"):
            if self.peek() == "AND":
                self.i += 1
            right = self.p_not()
            left = right if left is None else left if right is None else left & right
        return left

    def p_not(self):
        if self.peek() == "NOT":
            self.i += 1
            x = self.p_not()
            return None if x is None else set(range(len(self.docs))) - x
        return self.p_atom()

    def p_atom(self):
        t = self.peek()
        if t is None:
            return None
        self.i += 1
        if t == "(":
            r = self.p_or()
            if self.peek() == ")":
                self.i += 1
            return r
        if t == ")":
            return None
        if t.startswith('"'):
            return self.phrase(t.strip('"'))
        return self.term(t)

    # ---------- public search ----------
    def search(self, q):
        self.notes, self.qstems = [], set()
        self.toks = re.findall(r'"[^"]*"|\(|\)|[^\s()"]+', q)
        self.i = 0
        res = None
        while self.i < len(self.toks):
            if self.peek() == ")":
                self.i += 1
                continue
            r = self.p_or()
            res = r if res is None else res if r is None else res & r
        if res is None:
            return [], self.notes
        N = len(self.docs)
        ranked = []
        for d in res:
            score = 0.0
            for s in self.qstems:
                tf = len(self.index[s].get(d, ()))
                if tf:
                    score += (1 + math.log10(tf)) * math.log10(1 + N / len(self.index[s]))
            ranked.append((score, d))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return ranked, self.notes


# --------------------------------------------------------------------------
# 5. GUI  (Box 1 = query + Search button, Box 2 = results)
# --------------------------------------------------------------------------
from tkinter import ttk

BG = "#FFFBEA"
CARD = "#FFFFFF"
LIGHT_YELLOW = "#FFF4C2"
PRIMARY_YELLOW = "#FFD21F"
GOLDEN_YELLOW = "#FFC107"
DARK_GOLD = "#D99A00"
TEXT = "#493000"
MUTED = "#8A6D3B"
BORDER = "#F0C64B"
FONT = "Segoe UI"


class App:
    EXAMPLES = [
        ("Boolean", "cat AND NOT dog"),
        ("Phrase", '"machine learning"'),
        ("Wildcard", "comp*"),
        ("Soundex", "soundex:smith"),
        ("Spell check", "informaton"),
    ]

    def __init__(self, root):
        self.engine = SearchEngine()
        self.root = root
        root.title("Text Search Engine")
        root.geometry("1000x740")
        root.minsize(780, 580)
        root.configure(bg=BG)
        self._styles()

        # ---- header (centered) ----
        hdr = tk.Frame(root, bg=BG)
        hdr.pack(fill="x", pady=(20, 4))
        tk.Label(hdr, text="Text Search Engine", font=(FONT, 24, "bold"),
                 bg=BG, fg=TEXT).pack()
        tk.Label(hdr, text="Tokenization  ·  Stemming  ·  Inverted Index  ·  Wildcard  ·  "
                           "Soundex  ·  Spell Correction",
                 font=(FONT, 10), bg=BG, fg=MUTED).pack(pady=(3, 0))

        # ---- load button (centered in the middle) ----
        mid = tk.Frame(root, bg=BG)
        mid.pack(pady=(14, 10))
        ttk.Button(mid, text="Load .txt File", style="Accent.TButton",
                   command=self.load, cursor="hand2").pack()
        self.status = tk.Label(mid, text="No file loaded yet", font=(FONT, 10),
                               bg=BG, fg=MUTED)
        self.status.pack(pady=(8, 0))

        # ---- Box 1: query ----
        b1 = self._card(root, "BOX 1  ·  QUERY")
        b1.pack(fill="x", padx=24, pady=(4, 8))
        row = tk.Frame(b1, bg=CARD)
        row.pack(fill="x", padx=14, pady=(2, 6))
        self.query = tk.Entry(row, font=(FONT, 13), relief="flat", bg="#FFFFFF", fg=TEXT,
                              insertbackground=TEXT, highlightthickness=2,
                              highlightbackground=BORDER, highlightcolor=DARK_GOLD)
        self.query.pack(side="left", fill="x", expand=True, ipady=8)
        self.query.bind("<Return>", lambda e: self.search())
        ttk.Button(row, text="Clear", style="Ghost.TButton", command=self.clear,
                   cursor="hand2").pack(side="right", padx=(8, 0))
        ttk.Button(row, text="Search", style="Search.TButton", command=self.search,
                   cursor="hand2").pack(side="right", padx=(10, 0))

        chips = tk.Frame(b1, bg=CARD)
        chips.pack(pady=(0, 12))
        tk.Label(chips, text="Try:", font=(FONT, 9), bg=CARD, fg=MUTED).pack(side="left", padx=(0, 6))
        for label, q in self.EXAMPLES:
            self._chip(chips, label, q).pack(side="left", padx=3)

        # ---- Box 2: results ----
        b2 = self._card(root, "BOX 2  ·  RESULTS")
        b2.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self.out = scrolledtext.ScrolledText(
            b2, wrap="word", font=("Consolas", 11), relief="flat", bg="#FFFFFF", fg=TEXT,
            padx=14, pady=10, borderwidth=0, highlightthickness=1, highlightbackground=BORDER)
        self.out.pack(fill="both", expand=True, padx=14, pady=(2, 14))
        self.out.tag_config("hit", background=PRIMARY_YELLOW, foreground=TEXT)
        self.out.tag_config("docid", font=("Consolas", 11, "bold"), foreground=DARK_GOLD)
        self.out.tag_config("count", font=(FONT, 12, "bold"), foreground=TEXT)
        self.out.tag_config("note", foreground="#8C5E00")
        self.out.tag_config("muted", foreground=MUTED)
        self._show([("Load a text file, then type a query above.\n", "muted"),
                    ("Supports boolean, phrase, wildcard, Soundex and spell-corrected queries.\n", "muted")])

    # ---------- UI helpers ----------
    def _styles(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("Accent.TButton", background=PRIMARY_YELLOW, foreground=TEXT,
                    font=(FONT, 12, "bold"), padding=(30, 11), borderwidth=1,
                    bordercolor=BORDER, lightcolor=PRIMARY_YELLOW, darkcolor=PRIMARY_YELLOW,
                    focusthickness=0)
        s.map("Accent.TButton",
              background=[("pressed", DARK_GOLD), ("active", GOLDEN_YELLOW)],
              foreground=[("pressed", TEXT), ("active", TEXT)],
              bordercolor=[("pressed", DARK_GOLD), ("active", DARK_GOLD)])
        s.configure("Search.TButton", background=GOLDEN_YELLOW, foreground=TEXT,
                    font=(FONT, 11, "bold"), padding=(22, 8), borderwidth=1,
                    bordercolor=BORDER, lightcolor=GOLDEN_YELLOW, darkcolor=GOLDEN_YELLOW,
                    focusthickness=0)
        s.map("Search.TButton",
              background=[("pressed", DARK_GOLD), ("active", PRIMARY_YELLOW)],
              foreground=[("pressed", TEXT), ("active", TEXT)],
              bordercolor=[("pressed", DARK_GOLD), ("active", DARK_GOLD)])
        s.configure("Ghost.TButton", background=LIGHT_YELLOW, foreground=TEXT,
                    font=(FONT, 11), padding=(16, 8), borderwidth=1,
                    bordercolor=BORDER, lightcolor=LIGHT_YELLOW, darkcolor=LIGHT_YELLOW,
                    focusthickness=0)
        s.map("Ghost.TButton",
              background=[("pressed", BORDER), ("active", PRIMARY_YELLOW)],
              foreground=[("pressed", TEXT), ("active", TEXT)],
              bordercolor=[("pressed", DARK_GOLD), ("active", DARK_GOLD)])

    def _card(self, parent, title):
        card = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        tk.Label(card, text=title, font=(FONT, 9, "bold"), bg=CARD, fg=DARK_GOLD,
                 anchor="w").pack(fill="x", padx=14, pady=(10, 6))
        return card

    def _chip(self, parent, label, q):
        b = tk.Button(parent, text=label, font=(FONT, 9), relief="flat", bd=0,
                      bg=LIGHT_YELLOW, fg=TEXT, activebackground=PRIMARY_YELLOW,
                      activeforeground=TEXT, padx=10, pady=3, cursor="hand2",
                      command=lambda: self._use_example(q))
        b.bind("<Enter>", lambda e: b.config(bg=PRIMARY_YELLOW))
        b.bind("<Leave>", lambda e: b.config(bg=LIGHT_YELLOW))
        return b

    def _show(self, parts):
        self.out.config(state="normal")
        self.out.delete("1.0", "end")
        for text, tag in parts:
            self.out.insert("end", text, tag)
        self.out.config(state="disabled")

    def _use_example(self, q):
        self.query.delete(0, "end")
        self.query.insert(0, q)
        self.search()

    def clear(self):
        self.query.delete(0, "end")
        self._show([("Cleared. Type a new query above.\n", "muted")])
        self.query.focus_set()

    # ---------- actions ----------
    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                n_docs, n_terms = self.engine.build(f.read())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        name = path.replace("\\", "/").split("/")[-1]
        self.status.config(text=f"{name}  —  {n_docs} documents, {n_terms} unique terms indexed",
                           fg="#7A5200")
        self._show([("File indexed successfully. Enter a query and press Search.\n", "note")])
        self.query.focus_set()

    def search(self):
        if not self.engine.docs:
            messagebox.showinfo("No data", "Load a .txt file first.")
            return
        q = self.query.get().strip()
        if not q:
            return
        ranked, notes = self.engine.search(q)
        parts = [(f"Query:  {q}\n", "muted")]
        for n in notes:
            parts.append(("•  " + n + "\n", "note"))
        parts.append((f"\n{len(ranked)} document(s) found\n", "count"))
        parts.append(("─" * 60 + "\n\n", "muted"))
        for score, d in ranked[:50]:
            parts.append((f"Document {d + 1}", "docid"))
            parts.append((f"    relevance {score:.3f}\n", "muted"))
            doc, last = self.engine.docs[d], 0
            for tok, _, s, e in tokenize(doc):
                if tok not in STOP and stem(tok) in self.engine.qstems:
                    parts.append((doc[last:s], ""))
                    parts.append((doc[s:e], "hit"))
                    last = e
            parts.append((doc[last:] + "\n\n", ""))
        self._show(parts)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
