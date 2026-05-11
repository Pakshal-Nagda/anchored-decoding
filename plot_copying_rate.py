import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_copying_results(json_filepath="copying_results.json"):
    """Reads the JSON results and generates a line plot."""
    
    if not os.path.exists(json_filepath):
        print(f"Error: Could not find {json_filepath}. Please run the calculation script first.")
        return

    # 1. Load the JSON data
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not data:
        print("The JSON file is empty.")
        return

    # 2. Convert to a Pandas DataFrame
    df = pd.DataFrame(data)
    
    # 3. Data Cleaning & Sorting
    # Try to convert k_value to numeric for correct numerical sorting on the x-axis (e.g., 2, 5, 10 instead of 10, 2, 5)
    df['k_value_num'] = pd.to_numeric(df['k_value'], errors='coerce')
    
    # Sort the dataframe so the lines plot sequentially from left to right
    df = df.sort_values(by=['k_value_num', 'k_value'])

    # 4. Set up the plot style
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", palette="tab10")
    
    # 5. Create the line plot
    # If your k-values are strictly numeric, use x='k_value_num'. Otherwise, use x='k_value' (categorical)
    ax = sns.lineplot(
        data=df, 
        x='k_value', 
        y='average_copying_rate', 
        hue='method_name',
        marker='o',      # Adds dots at each data point
        linewidth=2.5,   # Makes lines thicker
        markersize=8     # Makes the dots larger
    )

    # 6. Formatting the chart
    plt.title("Average Copying Rate by Method and K-Value", fontsize=16, pad=15)
    plt.xlabel("K-Value", fontsize=12, labelpad=10)
    plt.ylabel("Average Copying Rate", fontsize=12, labelpad=10)
    
    # Format the y-axis as percentages (e.g., 0.82 -> 82%)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Adjust legend position
    plt.legend(title="Method Name", title_fontsize='11', fontsize='10', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Automatically adjust subplot parameters to give specified padding (prevents legend cutoff)
    plt.tight_layout()

    # 7. Save and display
    output_filename = "copying_rate_plot.png"
    #plt.savefig(output_filename, dpi=300) # Saves a high-res image
    #print(f"Plot saved successfully to {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    plot_copying_results("hp2/copying_results.json")