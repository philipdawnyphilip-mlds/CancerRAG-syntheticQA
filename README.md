# CancerRAG-syntheticQA
#### Repository for generation and evaluation of synthetic questions for radiation oncology answers

### Files:
- src/scores_analysis.py
  - Summary stats and correlation heatmaps for Generated Score, RQUGE, and Human scores. 
- src/topics_analysis.py
  - Topic-level summaries and topic wise kendall correlations, agreement %, MAE and SD of absolute error.
- src/lex_diversity_analysis.py
  - Lexical richness (TTR, cTTR), correlation heatmaps, Zipf plots and Zipf exponent computation.
- src/qn_regeneration_analysis.py
  - Preprocessing for regenerated questions, merged scores, correlations and visualization.
- src/thresholding_analysis.py
  - ROC/AUC plot generation for thresholding LLM/RQUGE against human score.

### Data
- Excel input files used by scripts

### Usage
- Ensure dependencies (Python 3.8+): pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, lexicalrichness.

`pip install requirements.txt`

- Run scripts from the project root, e.g.:
  - python src/scores_analysis.py
  - python src/topics_analysis.py
  - python src/lex_diversity_analysis.py

