import pandas as pd
from evaluate import load

# Load the saved CSVs (created earlier)
led = pd.read_csv("led_table_15.csv")
peg = pd.read_csv("pegasus_table_15.csv")
t5  = pd.read_csv("t5_table_15.csv")

bertscore = load("bertscore")

def compute_bertscore(df, model_name):
    refs = df["reference_abstract"].tolist()
    cands = df["generated_summary"].tolist()
    res = bertscore.compute(predictions=cands, references=refs, lang="en")
    # Average across the 15 papers
    out = {
        "precision": round(sum(res["precision"]) / len(res["precision"]) * 100, 2),
        "recall":    round(sum(res["recall"])    / len(res["recall"])    * 100, 2),
        "f1":        round(sum(res["f1"])        / len(res["f1"])        * 100, 2),
    }
    return out

led_scores = compute_bertscore(led, "LED")
peg_scores = compute_bertscore(peg, "PEGASUS")
t5_scores  = compute_bertscore(t5,  "T5")

summary = pd.DataFrame([led_scores, peg_scores, t5_scores],
                       index=["LED", "PEGASUS", "T5"])

print("\nBERTScore (mean over 15 papers):")
print(summary)

# Save for Chapter IV, Section 4.2
summary.to_csv("bertscore_summary.csv", encoding="utf-8")
