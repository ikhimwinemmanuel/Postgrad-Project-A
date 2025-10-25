import pandas as pd
import matplotlib.pyplot as plt

# Load your saved ROUGE summary
df = pd.read_csv("rouge_summary.csv", index_col=0)

# Create grouped bar chart
ax = df.plot(kind="bar", figsize=(8,6))
plt.title("ROUGE Scores Across Models (15 Consistent Papers)", fontsize=14)
plt.ylabel("Score (%)", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()

# Save figure for report
plt.savefig("rouge_scores_bar.png", dpi=300)

plt.show()
