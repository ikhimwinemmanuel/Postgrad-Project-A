import pandas as pd
import matplotlib.pyplot as plt

# Load Top-K summary (must have index = model names)
df = pd.read_csv("topk_summary.csv", index_col=0)

# Plot
ax = df.plot(kind="bar", figsize=(7,6), legend=False)
plt.title("Top-K Sentence-Level Cosine Similarity (Cumulative over 15 Papers)", fontsize=14)
plt.ylabel("Cumulative Score", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=0)

# Annotate bars with values
for i, val in enumerate(df.iloc[:,0]):
    plt.text(i, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig("topk_summary_bar.png", dpi=300)
plt.show()
