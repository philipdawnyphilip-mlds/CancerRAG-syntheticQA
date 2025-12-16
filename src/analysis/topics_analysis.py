import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, cohen_kappa_score
from scipy.stats import spearmanr, pearsonr, kendalltau

def load_and_preprocess_data(file_path):
    """Load and preprocess the data."""
    df = pd.read_excel(file_path)
    return df[df.valid_answer != 0]

def analyze_scores_by_topic(df, score_column, analysis_name):
    """Generate summary statistics by topic."""
    summary = df.groupby(['manual_topic'], as_index=False)[score_column].agg({
        'count': 'count' if score_column == 'new_generated_score' else None,
        'mean': lambda x: np.round(np.mean(x), 2),
        'sd': lambda x: np.round(np.std(x), 2)
    }).dropna(axis=1)
    
    print(f"\n{'-'*50}")
    print(f"{analysis_name} Analysis by Topic:")
    print(f"{'-'*50}")
    print(summary.to_string(index=False))
    return summary

def compute_topic_correlations(df):


    corr_df = df.groupby('manual_topic', as_index=False)[
        ['new_generated_score', 'newgenqs_rquge_score', 'new_human_score']
    ].agg(lambda x: list(x))

    # Kendall correlations 
    def kendall(a, b):
        try:
            if len(a) <= 1 or len(b) <= 1:
                return np.nan
            return np.round(kendalltau(a, b)[0], 2)
        except Exception:
            return np.nan

    corr_df['llm_human_kendall'] = corr_df.apply(
        lambda row: kendall(row['new_generated_score'], row['new_human_score']), axis=1
    )
    corr_df['rquge_human_kendall'] = corr_df.apply(
        lambda row: kendall(row['newgenqs_rquge_score'], row['new_human_score']), axis=1
    )

    # Agreement metrics
    corr_df['llm_human_agreement'] = corr_df.apply(
        lambda row: np.round(np.mean(np.abs(np.array(row['new_generated_score']) - np.array(row['new_human_score'])) == 0) * 100, 2),
        axis=1
    )
    corr_df['llm_human_agreement_tol1'] = corr_df.apply(
        lambda row: np.round(np.mean(np.abs(np.array(row['new_generated_score']) - np.array(row['new_human_score'])) <= 1) * 100, 2),
        axis=1
    )

    # MAE and SD of absolute errors
    corr_df['mae_abs_err'] = corr_df.apply(
        lambda row: np.round(np.mean(np.abs(np.array(row['new_generated_score']) - np.array(row['new_human_score']))), 2),
        axis=1
    )
    corr_df['sd_abs_err'] = corr_df.apply(
        lambda row: np.round(np.std(np.abs(np.array(row['new_generated_score']) - np.array(row['new_human_score']))), 2),
        axis=1
    )

    # 
    print("\nTopic-wise correlations & error summary")
    print(corr_df[['manual_topic', 'llm_human_kendall', 'rquge_human_kendall', 'llm_human_agreement', 'llm_human_agreement_tol1', 'mae_abs_err', 'sd_abs_err']].head(10).to_string(index=False))

    return corr_df

def main():
    # File path
    file_path = r"topic_analysis_0325.xlsx"
    
    # Load data
    df = load_and_preprocess_data(file_path)
    
    # Perform analyses
    llm_summary = analyze_scores_by_topic(df, 'new_generated_score', 'LLM Scores')
    rquge_summary = analyze_scores_by_topic(df, 'newgenqs_rquge_score', 'RQUGE Scores')
    human_summary = analyze_scores_by_topic(df, 'new_human_score', 'Human Scores')

    # New: compute and return topic correlations/errors
    corr_df = compute_topic_correlations(df)
    # ...user can save or further process corr_df...

if __name__ == "__main__":
    main()







