import pandas as pd
import matplotlib.pyplot as plt

# 1) Load the BERTScore summary you saved earlier
# Expected columns: precision, recall, f1; index: LED, PEGASUS, T5
df = pd.read_csv("bertscore_summary.csv", index_col=0)

# 2) Create a grouped bar chart
ax = df.plot(kind="bar", figsize=(8, 6))
plt.title("BERTScore Across Models (15 Consistent Papers)", fontsize=14)
plt.ylabel("Score (%)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=0)
plt.ylim(0, 100)  # scores are percentages
plt.legend(title="Metric")  # precision / recall / f1
plt.tight_layout()

# 3) Save a high-res figure for your report
plt.savefig("bertscore_scores_bar.png", dpi=300)
plt.show()
