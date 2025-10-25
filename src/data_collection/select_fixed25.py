# -*- coding: utf-8 -*-
"""
select_fixed25.py
Usage:
    python scripts/select_fixed25.py <input_jsonl> <output_jsonl>
Selects a deterministic set of 25 records from the full dataset.
"""

import json
import random
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/select_fixed25.py <input_jsonl> <output_jsonl>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    # Load all records
    rows = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Keep only records with both abstract and introduction (should already be true)
    rows = [r for r in rows if r.get("abstract") and r.get("introduction")]

    # Deterministic shuffle: stable seed + stable sort first
    rows.sort(key=lambda r: r["arxiv_id"])
    random.seed(42)
    random.shuffle(rows)

    subset = rows[:25]

    with dst.open("w", encoding="utf-8") as w:
        for r in subset:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Selected {len(subset)} -> {dst}")

if __name__ == "__main__":
    main()
