import os
import json
import glob
import re
from collections import Counter

def get_ngram_counts(text, n=4):
    """Generates a frequency count of character n-grams."""
    # Lowercase to make the comparison case-insensitive
    text = str(text).lower()
    
    # If the text is shorter than n, treat the whole string as one chunk
    if len(text) < n:
        return Counter([text])
        
    # Generate overlapping chunks of length n
    ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
    return Counter(ngrams)

def calculate_copy_rate(output_text, input_text, n=4):
    """
    Calculates what percentage of the output's n-grams exist in the input.
    Returns a float between 0.0 and 1.0.
    """
    if not output_text or not input_text:
        return 0.0
        
    out_counts = get_ngram_counts(output_text, n)
    in_counts = get_ngram_counts(input_text, n)
    
    total_out_ngrams = sum(out_counts.values())
    if total_out_ngrams == 0:
        return 0.0
        
    # Find the intersection of multiset counts: min(out_count, in_count)
    overlap = sum((out_counts & in_counts).values())
    
    return overlap / total_out_ngrams

def process_folder(folder_path="."):
    """Processes all matching JSON files and generates a summary report."""
    
    # Matches <methodname>_<k-value>_hp_gen_2.json
    # Uses regex to safely extract methodname (even if it has underscores) and k-value
    file_pattern = os.path.join(folder_path, "*_*_hp_gen_2.json")
    regex_pattern = re.compile(r"^(.*?)_([^_]+)_hp_gen_2\.json$")
    
    results = []
    
    # Find all JSON files in the directory matching the pattern
    for filepath in glob.glob(file_pattern):
        filename = os.path.basename(filepath)
        match = regex_pattern.match(filename)
        
        if not match:
            continue
            
        method_name = match.group(1)
        k_value = match.group(2)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not data:
                continue
                
            # Calculate copying rate for each item in the file
            file_copy_rates = []
            for item in data:
                # You can change item.get("input") to item.get("reference") if needed
                input_text = item.get("input", "") 
                output_text = item.get("output", "")
                
                rate = calculate_copy_rate(output_text, input_text)
                file_copy_rates.append(rate)
                
            # Calculate the average for this specific file
            avg_rate = sum(file_copy_rates) / len(file_copy_rates) if file_copy_rates else 0.0
            
            # Store the aggregate data
            results.append({
                "file": filename,
                "method_name": method_name,
                "k_value": k_value,
                "average_copying_rate": round(avg_rate, 4), # Rounded for readability
                "total_records_processed": len(file_copy_rates)
            })
            
            print(f"Processed {filename}: Avg Rate = {avg_rate:.2%}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Write the final summarized data to the new JSON file
    output_filepath = os.path.join(folder_path, "copying_results.json")
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nDone! Results saved to {output_filepath}")

if __name__ == "__main__":
    # Ensure you set this to your folder path, or leave as "." if running in the same directory
    FOLDER_PATH = "hp2" 
    process_folder(FOLDER_PATH)
