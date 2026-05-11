import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

def load_and_merge_data(copy_json_path, mauve_json_path):
    """Loads both JSON files, standardizes them, and merges into one DataFrame."""
    
    if not os.path.exists(copy_json_path) or not os.path.exists(mauve_json_path):
        raise FileNotFoundError("Make sure both copying_results.json and the mauve JSON file exist in the directory.")

    # 1. Load Data
    df_copy = pd.read_json(copy_json_path)
    df_mauve = pd.read_json(mauve_json_path)

    # 2. Standardize column names so they match for the merge
    # In the previous script, we used 'method_name' and 'k_value'. 
    # The new JSON uses 'method' and 'k'.
    if 'method_name' in df_copy.columns:
        df_copy.rename(columns={'method_name': 'method', 'k_value': 'k'}, inplace=True)
        
    # 3. Ensure 'k' is a float in both dataframes so they merge perfectly
    df_copy['k'] = df_copy['k'].astype(float)
    df_mauve['k'] = df_mauve['k'].astype(float)

    # 4. Merge the dataframes on Method and K
    df_merged = pd.merge(df_copy, df_mauve, on=['method', 'k'], how='inner')
    
    # Sort for clean line plotting
    df_merged = df_merged.sort_values(by=['method', 'k'])
    
    return df_merged

def plot_tradeoff_curve(df):
    """Plots MAUVE vs Copying Rate. This is standard in NLP generation research."""
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")

    # Plot the trade-off lines
    ax = sns.lineplot(
        data=df, 
        x='average_copying_rate', 
        y='mauve', 
        hue='label',  # Using 'label' from the Mauve JSON for prettier names
        marker='o',
        linewidth=2.5,
        markersize=9
    )

    # Annotate the k-values next to the points so we know which point corresponds to which k
    for i in range(df.shape[0]):
        plt.text(
            df['average_copying_rate'].iloc[i] + 0.005, # slight X offset
            df['mauve'].iloc[i],
            f"k={df['k'].iloc[i]}",
            fontsize=9,
            color='gray'
        )

    plt.title("Quality vs. Copying Trade-off by Decoding Method", fontsize=16, pad=15)
    plt.xlabel("Average Copying Rate (Extractiveness)", fontsize=13)
    plt.ylabel("MAUVE Score (Fluency/Quality)", fontsize=13)
    
    # Format X axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    #plt.savefig("tradeoff_curve.png", dpi=300)
    print("Trade-off plot saved to 'tradeoff_curve.png'")
    plt.show()

def plot_dual_axis_by_method(df):
    """Creates separate subplots for each method showing K vs MAUVE and Copying Rate."""
    methods = df['label'].unique()
    n_methods = len(methods)
    
    # Create a dynamic 1xN grid of subplots
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5.5))
    if n_methods == 1:
        axes = [axes] # Ensure it's iterable if there's only one method
        
    sns.set_theme(style="white")

    for idx, method_label in enumerate(methods):
        ax1 = axes[idx]
        method_data = df[df['label'] == method_label]
        
        # Plot MAUVE on the left Y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('K-Value', fontsize=12)
        ax1.set_ylabel('MAUVE Score', color=color1, fontsize=12)
        ax1.plot(method_data['k'], method_data['mauve'], color=color1, marker='o', linewidth=2.5)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Create a second Y-axis sharing the same X-axis
        ax2 = ax1.twinx()  
        color2 = 'tab:red'
        ax2.set_ylabel('Copying Rate', color=color2, fontsize=12)
        ax2.plot(method_data['k'], method_data['average_copying_rate'], color=color2, marker='s', linewidth=2.5, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Format the right axis as a percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

        ax1.set_title(f"Method: {method_label}", fontsize=14, pad=10)
        ax1.grid(True, alpha=0.3)

    plt.suptitle("Impact of K-Value on MAUVE and Copying Rate per Method", fontsize=16, y=1.05)
    plt.tight_layout()
    #plt.savefig("metrics_by_method_dual_axis.png", dpi=300, bbox_inches='tight')
    print("Individual method subplots saved to 'metrics_by_method_dual_axis.png'")
    plt.show()

if __name__ == "__main__":
    # Ensure these paths point to your actual JSON files
    COPY_FILE = "hp2/copying_results.json" 
    MAUVE_FILE = "mauve_hp_1.json" # <--- Rename this to whatever your second JSON is named
    
    try:
        merged_df = load_and_merge_data(COPY_FILE, MAUVE_FILE)
        
        print(f"Successfully merged {len(merged_df)} data points.")
        
        # Generate the plots
        plot_tradeoff_curve(merged_df)
        plot_dual_axis_by_method(merged_df)
        
    except Exception as e:
        print(f"An error occurred: {e}")
