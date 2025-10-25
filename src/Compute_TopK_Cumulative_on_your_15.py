import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------- Config ---------
CSV_LED = "led_table_15.csv"
CSV_PEG = "pegasus_table_15.csv"
CSV_T5  = "t5_table_15.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, stable

# --------- Helpers ---------
def split_sentences(text: str):
    if not isinstance(text, str) or not text.strip():
        return []
    # Simple, deterministic segmentation on ., !, ?
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def paper_topk_mean(ref_text: str, gen_text: str, embedder: SentenceTransformer):
    """
    For a single paper:
      - build S (m x n) cosine matrix
      - k = min(m, n)
      - paper_score = mean of top-k similarities
    Returns (paper_score, m, n, k).
    """
    ref_sents = split_sentences(ref_text)
    gen_sents = split_sentences(gen_text)
    m, n = len(ref_sents), len(gen_sents)
    if m == 0 or n == 0:
        return 0.0, m, n, 0

    ref_emb = embedder.encode(ref_sents, convert_to_numpy=True, normalize_embeddings=True)
    gen_emb = embedder.encode(gen_sents, convert_to_numpy=True, normalize_embeddings=True)

    S = cosine_similarity(ref_emb, gen_emb)  # shape (m, n)
    k = min(m, n)
    flat = S.ravel()
    # top-k largest values (no need to fully sort)
    topk_vals = np.partition(flat, -k)[-k:]
    paper_score = float(np.mean(topk_vals))  # <-- mean over k top similarities
    return paper_score, m, n, k

def evaluate_model(df: pd.DataFrame, model_label: str, embedder: SentenceTransformer) -> pd.DataFrame:
    """
    Compute paper_score (TopK mean) for each of the 15 rows in df.
    Returns a dataframe with columns: arxiv_id, title, paper_score, m, n, k, model
    """
    rows = []
    for arxiv_id, title, ref, gen in zip(df["arxiv_id"], df["title"], df["reference_abstract"], df["generated_summary"]):
        score, m, n, k = paper_topk_mean(ref, gen, embedder)
        rows.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "paper_score_topk_mean": score,
            "ref_sent_count_m": m,
            "gen_sent_count_n": n,
            "k_min_mn": k,
            "model": model_label
        })
    return pd.DataFrame(rows)

# --------- Main ---------
if __name__ == "__main__":
    # Load your fixed 15-paper tables
    led_df = pd.read_csv(CSV_LED)
    peg_df = pd.read_csv(CSV_PEG)
    t5_df  = pd.read_csv(CSV_T5)

    # Load embedder once
    embedder = SentenceTransformer(EMBED_MODEL)

    # Per-paper evaluations
    led_eval = evaluate_model(led_df, "LED", embedder)
    peg_eval = evaluate_model(peg_df, "PEGASUS", embedder)
    t5_eval  = evaluate_model(t5_df,  "T5", embedder)

    # Save per-paper (useful for appendix/audit)
    led_eval.to_csv("topk_per_paper_LED.csv", index=False, encoding="utf-8")
    peg_eval.to_csv("topk_per_paper_PEGASUS.csv", index=False, encoding="utf-8")
    t5_eval.to_csv("topk_per_paper_T5.csv", index=False, encoding="utf-8")

    # ---- Model-level cumulative sums (YOUR FINAL VALUES) ----
    summary = pd.DataFrame({
        "TopK_cumulative_sum": [
            float(led_eval["paper_score_topk_mean"].sum()),
            float(peg_eval["paper_score_topk_mean"].sum()),
            float(t5_eval["paper_score_topk_mean"].sum())
        ]
    }, index=["LED", "PEGASUS", "T5"])

    print("\nCumulative Top-K (sum of 15 per-paper means) per model:")
    print(summary.round(4))

    # Save for Chapter IV, Section 4.3 (Table 4.3d)
    summary.round(4).to_csv("topk_summary.csv", encoding="utf-8")
