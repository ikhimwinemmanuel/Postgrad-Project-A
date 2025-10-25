# -*- coding: utf-8 -*-
"""
run_summarize.py
Usage:
  python scripts/run_summarize.py --model_name t5-large \
    --input data/processed/fixed25.jsonl \
    --output outputs/t5_large.jsonl
Supported examples:
  --model_name t5-large
  --model_name google/pegasus-xsum
  --model_name allenai/led-base-16384
"""

import argparse, json, time, os, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_data(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def pick_lengths(model_name: str):
    # safe, simple defaults
    model = model_name.lower()
    if "t5" in model:
        return 1024, 200   # input, output tokens
    if "pegasus" in model:
        return 512, 128
    if "led" in model:
        return 4096, 256
    return 1024, 200

def build_input_text(model_name: str, intro: str):
    if "t5" in model_name.lower():
        return "summarize: " + intro
    return intro

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=1)  # keep 1 to keep memory simple/comparable
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    print(f"[info] loading model: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    max_inp, max_out = pick_lengths(args.model_name)
    print(f"[info] token limits: max_input={max_inp}, max_output={max_out}")

    rows = load_data(inp)
    print(f"[info] loaded {len(rows)} records from {inp}")

    # generation config (simple, deterministic-ish)
    gen_kwargs = dict(
        max_new_tokens=max_out,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    saved = 0
    t0_all = time.time()
    with outp.open("w", encoding="utf-8") as w:
        for i, r in enumerate(rows, 1):
            intro = (r.get("introduction") or "").strip()
            if not intro:
                continue

            text = build_input_text(args.model_name, intro)

            # tokenize with truncation
            enc = tok(
                text,
                max_length=max_inp,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k,v in enc.items()}

            torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
            t0 = time.time()
            with torch.no_grad():
                out_ids = model.generate(**enc, **gen_kwargs)
            dt = time.time() - t0
            max_mem = torch.cuda.max_memory_allocated() if device == "cuda" else 0

            summary = tok.decode(out_ids[0], skip_special_tokens=True)

            rec = {
                "arxiv_id": r.get("arxiv_id"),
                "title": r.get("title"),
                "reference_abstract": r.get("abstract"),
                "generated_summary": summary,
                "model_name": args.model_name,
                "time_sec": round(dt, 3),
                "gpu_mem_bytes": int(max_mem),
                "input_tokens": int(enc["input_ids"].shape[-1]),
                "output_tokens": int(out_ids.shape[-1]),
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved += 1

            if i % 5 == 0 or i == len(rows):
                print(f"[progress] {i}/{len(rows)} processed, {saved} saved")

    print(f"[done] wrote {saved} records -> {outp} | total_time={round(time.time()-t0_all,1)}s")

if __name__ == "__main__":
    main()
