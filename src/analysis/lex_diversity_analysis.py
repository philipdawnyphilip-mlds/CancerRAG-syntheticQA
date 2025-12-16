# ! pip install lexicalrichness

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, cm
import seaborn as sns
import numpy as np
import math
import re
from collections import Counter
from lexicalrichness import LexicalRichness
from scipy.stats import linregress, pearsonr, spearmanr, kendalltau




def load_data(filepath):
    df = pd.read_excel(filepath)
    
    # Moving old and new gen Qs into single column
    df.new_generated_question = df.new_generated_question.fillna(df['Generated Question'])
    return df[df.valid_answer != 0]


def calculate_ttr(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    words = text.split()
    types = set(words)
    if len(words) == 0:
        return 0
    return len(types) / len(words)


def corrected_ttr(text):
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    tokens = text.split()
    num_tokens = len(tokens)
    num_types = len(set(tokens))
    if num_tokens == 0:
        return 0.0
    return num_types / math.sqrt(2 * num_tokens)



def calculate_lexical_metrics(df):
    df['question_ttr'] = df['new_generated_question'].apply(calculate_ttr)
    df['answer_ttr'] = df['Answer'].apply(calculate_ttr)
    df['question_cttr'] = df['new_generated_question'].apply(corrected_ttr)
    df['answer_cttr'] = df['Answer'].apply(corrected_ttr)
    return df




def create_correlation_heatmap(df, method='spearman', output_path=None):
    cols = ['question_cttr', 'answer_cttr', 'new_generated_score', 'new_human_score']
    label_map = {
        'new_human_score': 'Human Score',
        'new_generated_score': 'LLM Score',
        'answer_cttr': 'Answer CTTR',
        'question_cttr': 'Question CTTR'
    }
    
    corr_matrix = df[cols].corr(method=method)
    pval_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    
    for i in cols:
        for j in cols:
            if i == j:
                pval_matrix.loc[i, j] = 0.0
            else:
                _, pval = spearmanr(df[i], df[j])
                pval_matrix.loc[i, j] = pval

    annot_labels = corr_matrix.round(2).astype(str) + '\n(p=' + pval_matrix.round(3).astype(str) + ')'
    
    mpl.rc('font', family='Times New Roman')
    plt.figure(figsize=(10, 8), dpi=250)
    
    heatmap = sns.heatmap(
        corr_matrix,
        vmin=-1, vmax=1,
        annot=annot_labels,
        fmt='',
        linewidths=0.5,
        linecolor='white',
        cmap='Purples',
        cbar=True,
        annot_kws={"size": 18, "weight": "bold", "color": "#444444"},
        square=True,
        norm=colors.PowerNorm(gamma=1.5, vmin=-1, vmax=1),
        alpha=0.45
    )

    # Set background and grid
    heatmap.set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')

    # Tweak tick labels for a cleaner look
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=13, fontweight='bold', color='#444444')
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=13, fontweight='bold', color='#444444')

    # Remove top and right spines for minimalist style
    sns.despine(left=True, bottom=True)

    # Style the colorbar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15, colors='#444444')
    cbar.outline.set_visible(False)


    new_labels = [label_map.get(col, col) for col in corr_matrix.columns]
    heatmap.set_xticklabels(new_labels, fontsize=13, fontweight='bold', color='#444444')
    heatmap.set_yticklabels(new_labels, fontsize=13, fontweight='bold', color='#444444')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()



def plot_zipf_distribution(text, title="Zipf Distribution", output_path=None):
    plt.rcParams["font.family"] = "Times New Roman"
    words = text.lower().split()
    word_counts = Counter(words)
    frequencies = [count for word, count in word_counts.most_common()]
    ranks = range(1, len(frequencies) + 1)

    pastel_color = cm.get_cmap("Pastel1")(0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=250)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(ranks, frequencies, marker='o', markersize=5, color=pastel_color, linewidth=2.5, label=None)
    
    # Log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Minimal, pastel-style grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#e0e0e0', alpha=0.7)

    # Labels and title
    ax.set_xlabel('Rank', fontsize=16, fontweight='bold', color='#444444')
    ax.set_ylabel('Frequency', fontsize=16, fontweight='bold', color='#444444')
    ax.set_title(title, fontsize=18, fontweight='bold', color='#444444', pad=12)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ticks styling
    ax.tick_params(axis='both', labelsize=15, colors='#444444')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()



def compute_zipf_exponent(text):
    words = text.lower().split()
    word_counts = Counter(words)
    frequencies = [count for word, count in word_counts.most_common()]
    ranks = range(1, len(frequencies) + 1)
    log_ranks = np.log(ranks)
    log_freqs = np.log(frequencies)
    slope, intercept, r_value, p_value, std_err = linregress(log_ranks, log_freqs)
    return -slope



def main():
    # Load and process data
    data_path = r"topic_analysis_0325.xlsx"
    output_dir = "plots"
    
    df = load_data(data_path)
    df = calculate_lexical_metrics(df)
    
    # Create correlation heatmaps
    create_correlation_heatmap(df, method='spearman', 
                             output_path=f"{output_dir}/spearman_correlation.png")
    create_correlation_heatmap(df, method='kendall', 
                             output_path=f"{output_dir}/kendall_correlation.png")
    
    # Prepare and plot Zipf distributions
    qs_combined = re.sub(r'[^A-Za-z0-9\s]', '', ' '.join(df.new_generated_question.str.lower().to_list()))
    ans_combined = re.sub(r'[^A-Za-z0-9\s]', '', ' '.join(df.Answer.str.lower().to_list()))
    
    plot_zipf_distribution(qs_combined, "Questions Zipf Distribution", 
                          f"{output_dir}/questions_zipf.png")
    plot_zipf_distribution(ans_combined, "Answers Zipf Distribution", 
                          f"{output_dir}/answers_zipf.png")


if __name__ == "__main__":
    main()