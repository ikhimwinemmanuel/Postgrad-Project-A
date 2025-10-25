import pandas as pd
import evaluate

# Load CSVs created earlier
led_table_full = pd.read_csv("led_table_15.csv")
pegasus_table_full = pd.read_csv("pegasus_table_15.csv")
t5_table_full = pd.read_csv("t5_table_15.csv")

# Load ROUGE
rouge = evaluate.load("rouge")

def compute_rouge(df, model_name):
    refs = df["reference_abstract"].tolist()
    cands = df["generated_summary"].tolist()
    results = rouge.compute(predictions=cands, references=refs, use_stemmer=True)
    scores = {k: round(v * 100, 2) for k, v in results.items()}  # percentages
    return scores

# Run for each model
led_scores = compute_rouge(led_table_full, "LED")
pegasus_scores = compute_rouge(pegasus_table_full, "PEGASUS")
t5_scores = compute_rouge(t5_table_full, "T5")

# Build summary DataFrame
rouge_summary = pd.DataFrame(
    [led_scores, pegasus_scores, t5_scores],
    index=["LED", "PEGASUS", "T5"]
)

print("\nSummary ROUGE Table:")
print(rouge_summary)

# ---- Save results as CSV ----
rouge_summary.to_csv("rouge_summary.csv", index=True, encoding="utf-8")

