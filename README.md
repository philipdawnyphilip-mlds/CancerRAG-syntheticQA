CancerRAG-syntheticQA
A small collection of analysis scripts and notebooks for evaluating LLM-generated QA pairs (scores, lexical diversity, topic analysis, thresholding).

Files (brief)
- src/scores_analysis.py
  - Summary stats and correlation heatmaps for Generated Score, RQUGE, and Human scores. Uses correlation_analysis(...) to print kendall/spearman and plot heatmaps.
- src/scores_analysis_0306.ipynb
  - Exploratory notebook with step-by-step correlation tests, plotting, and Wilcoxon tests.
- src/topics_analysis.py
  - Topic-level summaries and compute_topic_correlations(...) for per-topic kendall, agreement %, MAE and SD of abs error.
- src/topics_analysis.ipynb
  - Notebook version of the topic analysis and visualization.
- src/lex_diversity_analysis.py
  - Lexical richness (TTR, cTTR), correlation heatmaps, Zipf plots and Zipf exponent computation.
- src/qn_regeneration_analysis.py
  - Preprocessing for regenerated questions, merged scores, correlation printing and stacked bar chart generation.
- src/thresholding_analysis.py
  - ROC/AUC plot generation for thresholding RQUGE against binary human score.

Data
- Put the Excel input files used by scripts in a data/ folder (or update paths in the scripts).
- Typical files:
  - qafinal_foranalysis_0323.xlsx
  - topic_analysis_0325.xlsx
  - scores_forthres_0324.xlsx
  - newqs_rqugegen_0323.xlsx
  - QA_pairs_final_review_02_11_reviewedTroy.xlsx

Usage
- Ensure dependencies (Python 3.8+): pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, lexicalrichness.
- Run scripts from the project root, e.g.:
  - python src/scores_analysis.py
  - python src/topics_analysis.py
  - python src/lex_diversity_analysis.py

Notes
- Scripts expect Excel files at paths referenced inside each file; either place data files in the project root or update paths.
- Notebooks contain exploratory analysis and are useful for finer-grained inspection.

Requirements
- pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, lexicalrichness

