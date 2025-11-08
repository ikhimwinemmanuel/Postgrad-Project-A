# Leveraging Large Language Models for Summary/Abstract Extraction of Scientific Texts

This repository contains the code, processed data, and figures for my postgraduate project evaluating three abstractive summarisation hugging face models (**LED**, **PEGASUS**, and **T5**) on scientific papers from arXiv (cs.AI and cs.LG).
## Project Report
The full report (including methodology, results, and analysis) is available [here](docs/Leveraging_Large%20Language_Models_for_Summary_Abstract_Extraction_of_Scientific_Texts.pdf).



## Repository Structure
```text
POSTGRADUATE_PROJECT_A/
├─ data/
│  └─ processed/
│     ├─ led_table_15.csv
│     ├─ pegasus_table_15.csv
│     ├─ t5_table_15.csv
│     ├─ rouge_summary.csv
│     ├─ bertscore_summary.csv
│     ├─ topk_summary.csv
│     ├─ efficiency_summary.csv
│     ├─ topk_per_paper_LED.csv
│     ├─ topk_per_paper_PEGASUS.csv
│     └─ topk_per_paper_T5.csv
├─ docs/
│  ├─ README.md
│  └─ Leveraging_Large_Language_Models_for_Summary_Abstract_Extraction_of_Scientific_Texts.pdf
├─ figures/
│  ├─ rouge_scores_bar.png
│  ├─ bertscore_scores_bar.png
│  └─ topk_summary_bar.png
├─ src/
│  ├─ df_build_and_save_15.py
│  ├─ Compute_ROUGE_on_your_15.py
│  ├─ rouge_summary_plot.py
│  ├─ Compute_BERTScore_on_your_15.py
│  ├─ bertscore_summary_plot.py
│  ├─ Compute_TopK_Final.py
│  ├─ topk_summary_plot.py
│  ├─ Compute_Efficiency_Summary.py
│  └─ data_collection/
│     ├─ collect_arxiv.py
│     ├─ select_fixed25.py
│     └─ run_summary_with_HF_model.py
└─ README.md

## Usage / Reproducing Results

All commands below assume you are in the project root (`POSTGRADUATE_PROJECT_A`) and your virtual environment is active.

### 1. Build DataFrames (15 consistent papers)
This extracts the 15 consistent papers across LED, PEGASUS, and T5 and saves them as CSVs.
```powershell
python src/df_build_and_save_15.py

python src/Compute_ROUGE_on_your_15.py
python src/rouge_summary_plot.py

python src/Compute_BERTScore_on_your_15.py
python src/bertscore_summary_plot.py

python src/Compute_TopK_Final.py
python src/topk_summary_plot.py

python src/Compute_Efficiency_Summary.py


## Results

All processed results are saved in `data/processed/` and all figures are saved in `figures/`.

- **ROUGE**  
  - Table: `data/processed/rouge_summary.csv`  
  - Figure: `figures/rouge_scores_bar.png`  

- **BERTScore**  
  - Table: `data/processed/bertscore_summary.csv`  
  - Figure: `figures/bertscore_scores_bar.png`  

- **Sentence-Level Cosine Similarity (Top-K)**  
  - Tables:  
    - `data/processed/topk_summary.csv`  
    - `data/processed/topk_per_paper_LED.csv`  
    - `data/processed/topk_per_paper_PEGASUS.csv`  
    - `data/processed/topk_per_paper_T5.csv`  
  - Figure: `figures/topk_summary_bar.png`  

- **Efficiency Metrics**  
  - Table: `data/processed/efficiency_summary.csv`  

These correspond directly to **Chapter IV** in the report:  
- Table 4.1d / Figure 4.1 → ROUGE  
- Table 4.2d / Figure 4.2 → BERTScore  
- Table 4.3d / Figure 4.3 → Top-K Cosine Similarity  
- Table 4.4d → Efficiency Metrics  

@misc{ikhimwin2025summarisation,
  title     = {Leveraging Large Language Models for Summary/Abstract Extraction of Scientific Texts},
  author    = {Ikhimwin Emmanuel},
  year      = {2025},
  howpublished = {\url{https://github.com/ikhimwinemmanuel/Postgrad-Project-A}}
}

