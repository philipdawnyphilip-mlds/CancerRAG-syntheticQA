import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, cohen_kappa_score, accuracy_score, recall_score, precision_score
from scipy.stats import spearmanr, pearsonr, kendalltau
import scipy.stats as stats
from scipy.stats import chi2_contingency
import os

def load_and_preprocess_data():
    df = pd.read_excel(r"scores_forthres_0324.xlsx")
    
    # Merge old LLM scores
    df.new_generated_score = df.new_generated_score.fillna(df['Generated Score'])
    df.new_generated_reasoning = df.new_generated_reasoning.fillna(df['Generated Reasoning'])
    
    # Load and process human scores
    new_human_scores_df = pd.read_excel(r"QA_pairs_final_review_02_11_reviewedTroy.xlsx")
    
    for i, row in new_human_scores_df.iterrows():
        df.loc[df.new_generated_question == row.new_generated_question, 'new_human_score'] = row.new_human_score
    
    df.new_human_score = df.new_human_score.fillna(df.human_score)
    df = df[df.valid_answer != 0]
    
    # Load RQUGE scores
    new_rquge = pd.read_excel(r"newqs_rqugegen_0323.xlsx")
    new_rquge = new_rquge[new_rquge.valid_answer != 0]
    
    return df, new_rquge

def print_summary_statistics(df, new_rquge):
    print("\nSummary Statistics:")
    print("-" * 50)
    print("New LLM Scores:")
    print(df.new_generated_score.describe())
    print("\nNew Human Scores:")
    print(df.new_human_score.describe())
    print("\nNew RQUGE Scores:")
    print(new_rquge.dropna(subset=['new_generated_question']).newgenqs_rquge_score.describe())

def calculate_correlations(df, new_rquge):
    print("\nCorrelation Analysis:")
    print("-" * 50)
    
    # LLM vs RQUGE
    kendall_corr, kendall_p = kendalltau(df.new_generated_score, new_rquge.newgenqs_rquge_score)
    spearman_corr, spearman_p = spearmanr(df.new_generated_score, new_rquge.newgenqs_rquge_score)
    print("\nLLM vs RQUGE:")
    print(f"Kendall correlation: {np.round(kendall_corr, 3)}, p-value: {np.round(kendall_p, 3)}")
    print(f"Spearman correlation: {np.round(spearman_corr, 3)}, p-value: {np.round(spearman_p, 3)}")
    
    # LLM vs Human
    kendall_corr, kendall_p = kendalltau(df.new_generated_score, df.new_human_score)
    spearman_corr, spearman_p = spearmanr(df.new_generated_score, df.new_human_score)
    print("\nLLM vs Human:")
    print(f"Kendall correlation: {np.round(kendall_corr, 3)}, p-value: {np.round(kendall_p, 3)}")
    print(f"Spearman correlation: {np.round(spearman_corr, 3)}, p-value: {np.round(spearman_p, 3)}")
    
    # Human vs RQUGE
    kendall_corr, kendall_p = kendalltau(new_rquge.newgenqs_rquge_score, df.new_human_score)
    spearman_corr, spearman_p = spearmanr(new_rquge.newgenqs_rquge_score, df.new_human_score)
    print("\nHuman vs RQUGE:")
    print(f"Kendall correlation: {np.round(kendall_corr, 3)}, p-value: {np.round(kendall_p, 3)}")
    print(f"Spearman correlation: {np.round(spearman_corr, 3)}, p-value: {np.round(spearman_p, 3)}")

def create_stacked_bar_plot(plot_df):
    plt.rcParams["font.family"] = "Times New Roman"
    
    fig, ax = plt.subplots(figsize=(18, 8.5), dpi=250)
    ax.set_facecolor('white')
    fig.set_facecolor('white')

    # Prepare data for stacked bar chart
    score_value_levels = sorted(plot_df['Score Value'].unique())
    n_levels = len(score_value_levels)

    # Define the qualitative colormaps from matplotlib
    pastel1 = plt.cm.get_cmap('Pastel1_r')
    pastel2 = plt.cm.get_cmap('Pastel1_r')

    # Map groups to colormaps
    group_order = ["LLM Score", "Updated LLM Score", "Human Score", "Updated Human Score"]
    group_to_cmap = {
        "LLM Score": pastel1,
        "Updated LLM Score": pastel1,
        "Human Score": pastel2,
        "Updated Human Score": pastel2
    }

    # Generate colors for each group and score value
    group_to_colors = {}
    for group in group_order:
        cmap = group_to_cmap[group]
        colors = [cmap(i/(n_levels-1)) for i in reversed(range(n_levels))]
        group_to_colors[group] = colors

    # Create stacked bars for each group
    for i, group in enumerate(reversed(group_order)):  # Reverse to match your original order
        left = 0  # Start position for each segment
        
        # Get data for this group
        group_data = plot_df[plot_df['Score Name'] == group]
        
        # For each score value, add a segment to the bar
        for score in score_value_levels:
            # Get count for this score
            count_data = group_data[group_data['Score Value'] == score]
            if not count_data.empty:
                count = count_data['Count'].values[0]
            else:
                count = 0
                
            # Skip if count is zero
            if count > 0:
                # Get color for this score
                color_idx = score_value_levels.index(score)
                color = group_to_colors[group][color_idx]
                
                # Plot the segment
                bar = ax.barh(i, count, left=left, height=0.7, color=color, edgecolor='white', linewidth=0.5)
                
                # Add annotation
                if count > 0:
                    # Calculate segment width to determine annotation strategy
                    segment_width = count

                    if segment_width < 7:
                        # For very small segments, place text outside to the right
                        pass
                    elif segment_width < 10:
                        # For small segments, use smaller font and center
                        ax.text(left + count/2, i, f"{np.round(count/200*100,2)}%", 
                                ha='center', va='center',
                                fontsize=13.5, color='#333333', 
                                fontweight='bold', fontname="Times New Roman")
                    else:
                        # For normal segments, use standard annotation
                        ax.text(left + count/2, i, f"{np.round(count/200*100,2)}%", 
                                ha='left', va='center',
                                fontsize=18, color='#333333', 
                                fontweight='bold', fontname="Times New Roman")

            # Update left position for next segment
            left += count

    # Customize the plot
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(reversed(group_order), fontsize=22, fontweight="bold", color="#222222", fontname="Times New Roman")
    ax.set_xlabel("Frequency", size=22, labelpad=18, fontweight="bold", color="#222222", fontname="Times New Roman")
    ax.set_ylabel("", size=22, fontweight="bold", color="#222222", fontname="Times New Roman")

    # Grid lines
    ax.grid(axis='x', linestyle='-', linewidth=0.5, color='#dddddd')
    ax.set_axisbelow(True)

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Set x-axis ticks
    ax.tick_params(axis='x', labelsize=22, colors="#222222")

    plt.subplots_adjust(top=0.85, right=1.45)

    # Create custom legend
    legend_ax = fig.add_axes([0.92, 0.82, 0.2, 0.13])
    legend_ax.axis('off')

    # Plot rectangles for legend with the new color scheme
    for i, (llm_color, human_color, score) in enumerate(zip(
        group_to_colors['LLM Score'], 
        group_to_colors['Human Score'], 
        score_value_levels)):
        # LLM color (left column)
        legend_ax.add_patch(mpatches.Rectangle((0.8, i), 0.3, 1, color=llm_color, ec='none'))
        # Human color (middle column)
        legend_ax.add_patch(mpatches.Rectangle((1.2, i), 0.3, 1, color=human_color, ec='none'))
        # Score value (right column)
        legend_ax.text(1.8, i+0.5, str(score), va='center', fontsize=14, color='#222222', 
                      fontweight='bold', fontname="Times New Roman")

    # Add column label
    legend_ax.text(1.8, n_levels - 5.65, "Score", ha='center', va='top', fontsize=12.8, 
                  fontweight='bold', fontname="Times New Roman")

    # Set limits and hide axes
    legend_ax.set_xlim(0, 4)
    legend_ax.set_ylim(0, n_levels)
    legend_ax.invert_yaxis()

    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

def prepare_plot_data(df):
    scores = ['Generated Score', 'new_generated_score', 'human_score', 'new_human_score']
    plot_data = []

    for score in scores:
        value_counts = df[score].value_counts().sort_index()
        for value, count in value_counts.items():
            plot_data.append({"Score Name": score, "Score Value": value, "Count": count})

    # Convert to DataFrame for Seaborn
    plot_df = pd.DataFrame(plot_data)

    plot_df['Score Name'] = plot_df['Score Name']\
    .apply(lambda x: {"Generated Score":"LLM Score", "human_score":"Human Score", "new_generated_score":"Updated LLM Score",\
                      "new_human_score":"Updated Human Score"}.get(x))
    
    return plot_df

def main():
    # Load and preprocess data
    df, new_rquge = load_and_preprocess_data()
    
    # Print summary statistics
    print_summary_statistics(df, new_rquge)
    
    # Calculate and print correlations
    calculate_correlations(df, new_rquge)
    
    # Prepare plot data and create visualization
    plot_df = prepare_plot_data(df)
    create_stacked_bar_plot(plot_df)

if __name__ == "__main__":
    main()