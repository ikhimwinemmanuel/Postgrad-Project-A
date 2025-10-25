import json
import pandas as pd

# Load JSONL helper 
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

# Load the three model files 
led_records = load_jsonl("led_cpu_25.jsonl")
pegasus_records = load_jsonl("pegasus_cpu_25.jsonl")
t5_records = load_jsonl("t5_large_cpu_test.jsonl")

# Build dicts by arxiv_id 
led_dict = {rec["arxiv_id"]: rec for rec in led_records}
pegasus_dict = {rec["arxiv_id"]: rec for rec in pegasus_records}
t5_dict = {rec["arxiv_id"]: rec for rec in t5_records}

# Helper to compute word overlap
def compute_overlap(ref, text):
    return len(set(ref.lower().split()) & set(text.lower().split()))

# Find consistent papers across all 3 models
consistent_papers = []
for rec in led_records:
    aid = rec["arxiv_id"]
    if aid in pegasus_dict and aid in t5_dict:
        ref = rec["reference_abstract"]
        led_overlap = compute_overlap(ref, rec["generated_summary"])
        peg_overlap = compute_overlap(ref, pegasus_dict[aid]["generated_summary"])
        t5_overlap = compute_overlap(ref, t5_dict[aid]["generated_summary"])
        if led_overlap > 5 and peg_overlap > 5 and t5_overlap > 5:
            consistent_papers.append(aid)

# Select the first 15 (exact same as I showed you before) 
top15_ids = consistent_papers[:15]

# Build DataFrame function 
def build_table(records_dict, ids):
    rows = []
    for aid in ids:
        rec = records_dict[aid]
        rows.append({
            "arxiv_id": rec["arxiv_id"],
            "title": rec["title"],
            "reference_abstract": rec["reference_abstract"],
            "generated_summary": rec["generated_summary"],
            "model_name": rec["model_name"],
            "time_sec": rec["time_sec"],
            "gpu_mem_bytes": rec["gpu_mem_bytes"],
            "input_tokens": rec["input_tokens"],
            "output_tokens": rec["output_tokens"]
        })
    return pd.DataFrame(rows)

# Build the three DataFrames 
led_table_full = build_table(led_dict, top15_ids)
pegasus_table_full = build_table(pegasus_dict, top15_ids)
t5_table_full = build_table(t5_dict, top15_ids)

# Now you can inspect them in VS Code 
print("LED table:\n", led_table_full.head(), "\n")
print("Pegasus table:\n", pegasus_table_full.head(), "\n")
print("T5 table:\n", t5_table_full.head(), "\n")

# ---- SAVE THE 15-PAPER TABLES AS CSV ----
led_table_full.to_csv("led_table_15.csv", index=False, encoding="utf-8")
pegasus_table_full.to_csv("pegasus_table_15.csv", index=False, encoding="utf-8")
t5_table_full.to_csv("t5_table_15.csv", index=False, encoding="utf-8")

# (Optional) Save the exact list of arxiv_ids used for reproducibility
pd.Series(top15_ids, name="arxiv_id").to_csv("top15_ids.csv", index=False, encoding="utf-8")
