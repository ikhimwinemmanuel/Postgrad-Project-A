# -*- coding: utf-8 -*-
"""
collect_arxiv_simple.py
- Simple arXiv (cs.AI, cs.LG) collector for arxiv v2.x
- Writes JSONL to ./data/processed/ and caches PDFs in ./data/raw/pdfs/
- Each record: arxiv_id, title, abstract, introduction, pdf_path, published, categories
"""

from __future__ import annotations
import json, re, time, os, contextlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests          # pip install requests
import arxiv             # pip install arxiv
import fitz as pymupdf   # pip install pymupdf

# -----------------------
# Config (edit if needed)
# -----------------------
NUM_RECORDS      = 1000
QUERY            = "cat:cs.AI OR cat:cs.LG"
PDF_TIMEOUT_S    = 60
PDF_RETRIES      = 3
PDF_RETRY_SLEEP  = 5
PAGE_SIZE_START  = 25
PAGE_SIZE_MIN    = 5
PAGE_SIZE_MAX    = 100
INTRO_SCAN_CAP   = 1200
API_DELAY_S      = 5

# -----------------------
# Paths (ALWAYS relative to current working directory)
# -----------------------
ROOT      = Path.cwd().resolve()                 # wherever you run the script
PDF_DIR   = ROOT / "data" / "raw" / "pdfs"
PROC_DIR  = ROOT / "data" / "processed"
PDF_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH  = PROC_DIR / f"arxiv_csAI_csLG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


# -----------------------
# Helpers
# -----------------------
def extract_introduction(full_text: str) -> Optional[str]:
    """Find 'Introduction' section heuristically and return its text."""
    lines = [ln.strip() for ln in full_text.splitlines()]
    pat_intro = re.compile(r"^\s*(\d+(\.\d+)*)?\s*introduction\s*$", re.I)
    pat_next  = re.compile(
        r"^\s*(\d+(\.\d+)*)?\s*(related work|background|method|methods|approach|model|"
        r"experiments|results|discussion|conclusion|conclusions)\s*$",
        re.I,
    )

    start = None
    for i, ln in enumerate(lines):
        if len(ln) <= 50 and pat_intro.match(ln):
            start = i + 1
            break
    if start is None:
        return None

    stop = len(lines)
    for j in range(start, min(start + INTRO_SCAN_CAP, len(lines))):
        if len(lines[j]) <= 70 and pat_next.match(lines[j]):
            stop = j
            break

    snippet = "\n".join(lines[start:stop]).strip()
    return snippet or None


def pdf_to_text_quiet(pdf_path: Path) -> Optional[str]:
    """Extract full text from a PDF. Suppress noisy MuPDF warnings."""
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            doc = pymupdf.open(pdf_path.as_posix())
            text = "".join(page.get_text() for page in doc)
            doc.close()
        return text
    except Exception as e:
        print(f"[PDF read] {pdf_path.name}: {e}")
        return None


def download_pdf(url: str, dest: Path) -> bool:
    """Download PDF with retries. Returns True on success, False if skipped."""
    if dest.exists():
        return True
    for attempt in range(1, PDF_RETRIES + 1):
        try:
            r = requests.get(url, timeout=PDF_TIMEOUT_S)
            if r.status_code == 200:
                dest.write_bytes(r.content)
                return True
            print(f"[PDF {dest.stem}] attempt {attempt}/{PDF_RETRIES} HTTP {r.status_code}")
        except Exception as e:
            print(f"[PDF {dest.stem}] attempt {attempt}/{PDF_RETRIES} error: {e}")
        time.sleep(PDF_RETRY_SLEEP)
    print(f"[PDF {dest.stem}] skipping after {PDF_RETRIES} failed attempts")
    return False


# -----------------------
# Main
# -----------------------
def main():
    page_size = PAGE_SIZE_START
    collected = 0
    new_pdfs  = 0

    print(f"[start] root={ROOT}")
    print(f"[start] pdf_dir={PDF_DIR}")
    print(f"[start] out_file={OUT_PATH}")

    with OUT_PATH.open("w", encoding="utf-8") as w:
        while collected < NUM_RECORDS:
            # Build page-limited search; Client handles paging internally
            search = arxiv.Search(
                query=QUERY,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
                max_results=page_size,
            )
            client = arxiv.Client(page_size=page_size, delay_seconds=API_DELAY_S, num_retries=5)

            try:
                yielded = False
                for paper in client.results(search):
                    yielded = True

                    arxiv_id = paper.entry_id.split("/")[-1]
                    pdf_path = PDF_DIR / f"{arxiv_id}.pdf"

                    if download_pdf(paper.pdf_url, pdf_path):
                        if not pdf_path.exists():
                            new_pdfs += 1
                    else:
                        continue  # skip this paper

                    full_text = pdf_to_text_quiet(pdf_path)
                    if not full_text:
                        continue

                    intro = extract_introduction(full_text)
                    rec = {
                        "arxiv_id": arxiv_id,
                        "title": paper.title,
                        "abstract": paper.summary,
                        "introduction": intro,
                        "pdf_path": pdf_path.as_posix(),
                        "published": str(paper.published),
                        "categories": paper.categories,
                    }

                    if rec["abstract"] and rec["introduction"]:
                        w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        collected += 1
                        if collected % 50 == 0:
                            print(f"[progress] {collected}/{NUM_RECORDS} -> {OUT_PATH}")
                            # flush to disk
                            w.flush()
                            os.fsync(w.fileno())
                        if collected >= NUM_RECORDS:
                            break

                # Adaptive paging: back off on empty page, grow on success
                if not yielded:
                    new_size = max(PAGE_SIZE_MIN, page_size // 2)
                    print(f"[backoff] empty page -> page_size {page_size} -> {new_size}")
                    page_size = new_size
                    time.sleep(API_DELAY_S)
                else:
                    page_size = min(PAGE_SIZE_MAX, page_size * 2)

            except arxiv.UnexpectedEmptyPageError as e:
                new_size = max(PAGE_SIZE_MIN, page_size // 2)
                print(f"[empty-feed] {e}; page_size {page_size} -> {new_size}")
                page_size = new_size
                time.sleep(API_DELAY_S)
                continue

    print(f"[done] saved={collected} records | file={OUT_PATH}")
    print(f"[note] PDFs cached in {PDF_DIR}")


if __name__ == "__main__":
    main()
