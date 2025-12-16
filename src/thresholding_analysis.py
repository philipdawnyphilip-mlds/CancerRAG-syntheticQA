import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import os

def load_and_preprocess_data(file_path):
    """Load and preprocess the data."""
    df = pd.read_excel(file_path)
    df = df[df.valid_answer != 0]
    df['human_score_binary'] = df.new_human_score.apply(lambda x: 1 if x>3 else 0)
    return df

def calculate_roc_metrics(df):
    """Calculate ROC curve metrics."""
    fpr, tpr, thresholds = roc_curve(df.human_score_binary, df.newgenqs_rquge_score)
    auc = roc_auc_score(df.human_score_binary, df.newgenqs_rquge_score)
    return fpr, tpr, thresholds, auc

def create_roc_plot(fpr, tpr, auc, output_path):
    """Create and save ROC curve plot."""
    # Set font globally
    plt.rcParams["font.family"] = "Times New Roman"

    # Choose pastel colors
    pastel_cmap = cm.get_cmap('Pastel1')
    roc_color = pastel_cmap(0)      # Soft pink
    diag_color = pastel_cmap(1)     # Soft blue
    marker_color = pastel_cmap(2)   # Soft green

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    # ROC curve
    ax.plot(fpr, tpr, color=roc_color, linewidth=2.5, label=f'AUC = {auc:.2f}')
    # Diagonal
    ax.plot([0, 1], [0, 1], linestyle='--', color=diag_color, linewidth=2)
    # Optimal threshold marker
    ax.scatter(fpr[14], tpr[14], marker='o', color=marker_color, s=80, edgecolor='gray',
              label=f'Optimal Threshold: {2.67:.2f}', zorder=5)

    # Labels and styling
    ax.set_xlabel('False Positive Rate', fontsize=14, color='#222222')
    ax.set_ylabel('True Positive Rate', fontsize=14, color='#222222')
    ax.legend(frameon=False, fontsize=12, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=12, colors='#222222')
    ax.grid(False)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set paths
    input_file = "scores_forthres_0324.xlsx"
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "roc_curve.png")

    # Process data and create plot
    df = load_and_preprocess_data(input_file)
    fpr, tpr, thresholds, auc = calculate_roc_metrics(df)
    create_roc_plot(fpr, tpr, auc, output_path)

if __name__ == "__main__":
    main()
