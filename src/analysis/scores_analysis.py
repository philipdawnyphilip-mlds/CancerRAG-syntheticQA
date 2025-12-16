import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns
from scipy.stats import spearmanr, chisquare, kendalltau, mode, wilcoxon

def load_and_preprocess_data(file_path):
    """Load and preprocess the scores data."""
    df = pd.read_excel(file_path)
    df.new_generated_score = df.new_generated_score.fillna(df['Generated Score'])
    df.new_generated_reasoning = df.new_generated_reasoning.fillna(df['Generated Reasoning'])
    return df

def print_summary_statistics(df):
    """Print summary statistics for all score types."""
    print(f"LLM Score summary stats (before regeneration):\n{df['Generated Score'].describe()}\n")
    print(f"LLM Score summary stats (after regeneration):\n{df.new_generated_score.describe()}\n")
    print(f"RQUGE Score summary stats:\n{df.rquge_score.describe()}\n")
    print(f"Human Score summary stats:\n{df.human_score.describe()}\n")

def correlation_analysis(df, correlation_method="kendall", save_path=None):
    """Compute and print Kendall and Spearman correlations for key pairs, and plot heatmap using specified method."""
    # Print Kendall correlations (with p-values)
    k_llm_rquge, k_p1 = kendalltau(df['Generated Score'], df['rquge_score'])
    k_llm_human, k_p2 = kendalltau(df['Generated Score'], df['human_score'])
    k_rquge_human, k_p3 = kendalltau(df['rquge_score'], df['human_score'])
    print("Kendall correlations:")
    print(f"  LLM Score vs RQUGE:     corr={k_llm_rquge:.3f}, p={k_p1:.3g}")
    print(f"  LLM Score vs Human:     corr={k_llm_human:.3f}, p={k_p2:.3g}")
    print(f"  RQUGE vs Human:         corr={k_rquge_human:.3f}, p={k_p3:.3g}")
    
    # Print Spearman correlations (with p-values)
    s_llm_rquge, s_p1 = spearmanr(df['Generated Score'], df['rquge_score'])
    s_llm_human, s_p2 = spearmanr(df['Generated Score'], df['human_score'])
    s_rquge_human, s_p3 = spearmanr(df['rquge_score'], df['human_score'])
    print("Spearman correlations:")
    print(f"  LLM Score vs RQUGE:     corr={s_llm_rquge:.3f}, p={s_p1:.3g}")
    print(f"  LLM Score vs Human:     corr={s_llm_human:.3f}, p={s_p2:.3g}")
    print(f"  RQUGE vs Human:         corr={s_rquge_human:.3f}, p={s_p3:.3g}")
    
    # Plot heatmap (preserve previous plotting behavior)
    mpl.rc('font', family='Times New Roman')
    plt.figure(figsize=(8, 6), dpi=250)
    
    heatmap = sns.heatmap(
        df[['Generated Score', 'rquge_score', 'human_score']].corr(correlation_method),
        vmin=0 if correlation_method == "spearman" else -1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        linecolor='white',
        cmap="Purples",
        cbar=True,
        annot_kws={"size": 18, "weight": "bold", "color": "#444444"},
        square=True,
        norm=colors.PowerNorm(gamma=0.75, vmin=0, vmax=1),
        alpha=0.45
    )
    
    heatmap.set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    heatmap.set_xticklabels(['LLM Score', 'RQUGE Score', 'Human Score'], 
                           fontsize=13, fontweight='bold', color='#444444')
    heatmap.set_yticklabels(['LLM Score', 'RQUGE Score', 'Human Score'], 
                           fontsize=13, fontweight='bold', color='#444444')
    
    sns.despine(left=True, bottom=True)
    
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15, colors='#444444')
    cbar.outline.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def main():
    # Load and preprocess data
    file_path = r"qafinal_foranalysis_0323.xlsx"
    df = load_and_preprocess_data(file_path)
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Plot and save correlation heatmaps (use new function name)
    print("Plotting Kendall correlation heatmap...")
    correlation_analysis(df, "kendall", "kendall_correlation_heatmap.png")
    
    print("Plotting Spearman correlation heatmap...")
    correlation_analysis(df, "spearman", "spearman_correlation_heatmap.png")

if __name__ == "__main__":
    main()


