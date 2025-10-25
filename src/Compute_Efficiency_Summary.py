import pandas as pd

# Inputs: the 15-paper tables you already saved
LED_CSV = "led_table_15.csv"
PEG_CSV = "pegasus_table_15.csv"
T5_CSV  = "t5_table_15.csv"

def load(name):
    df = pd.read_csv(name)
    # Defensive cast in case any field came in as string
    for col in ["time_sec", "gpu_mem_bytes", "input_tokens", "output_tokens"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def summarize(df):
    """Return averages in the shape your report expects."""
    avg_time = df["time_sec"].mean()
    avg_mem_mb = (df["gpu_mem_bytes"].max() / (1024**2)) if "gpu_mem_bytes" in df else None
    # ^ peak memory across 15 runs is more informative than mean for GPU footprint
    avg_in_tok = df["input_tokens"].mean()
    avg_out_tok = df["output_tokens"].mean()
    return {
        "Avg_Runtime_sec": round(float(avg_time), 2),
        "Peak_GPU_Memory_MB": round(float(avg_mem_mb), 1) if avg_mem_mb is not None else None,
        "Avg_Input_Tokens": round(float(avg_in_tok), 1),
        "Avg_Output_Tokens": round(float(avg_out_tok), 1),
    }

# Load
led = load(LED_CSV)
peg = load(PEG_CSV)
t5  = load(T5_CSV)

# Summaries
led_sum = summarize(led)
peg_sum = summarize(peg)
t5_sum  = summarize(t5)

# Combine + save
eff_summary = pd.DataFrame([led_sum, peg_sum, t5_sum], index=["LED", "PEGASUS", "T5"])
print("\nEfficiency Summary (15 papers):")
print(eff_summary)

eff_summary.to_csv("efficiency_summary.csv", encoding="utf-8")
