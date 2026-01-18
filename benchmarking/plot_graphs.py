import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# Configuration
BENCHMARK_DIR = 'benchmarking'
OUTPUT_DIR = '.'

def get_version_number(filename):
    """Extracts version number 'v1' from 'benchmark_v1.csv' for sorting"""
    match = re.search(r'v(\d+)', filename)
    return int(match.group(1)) if match else 999

def print_report_card(version, df):
    """Prints a summary similar to benchmarking.py"""
    queries_proc = len(df)
    avg_lat = df['duration'].mean()
    mean_p10 = df['precision@10'].mean()
    
    # Calculate Grade Score (Harmonic Mean of P@5 and F1@30) - logic from benchmarking.py
    # We use a safe safe harmonic mean calc to avoid div by zero if columns exist
    if 'precision@5' in df.columns and 'f1@30' in df.columns:
        # Create temp series for calculation
        p5 = df['precision@5']
        f1 = df['f1@30']
        # Avoid division by zero
        safe_p5 = p5.replace(0, 0.0001)
        safe_f1 = f1.replace(0, 0.0001)
        grade_scores = (2 * safe_p5 * safe_f1) / (safe_p5 + safe_f1)
        mean_grade = grade_scores.mean()
    else:
        mean_grade = 0.0

    print("="*40)
    print(f"        REPORT CARD: {version}")
    print("="*40)
    print(f"Queries Processed:    {queries_proc}")
    print(f"Average Latency:      {avg_lat:.4f}s")
    print(f"Mean Precision@10:    {mean_p10:.4f}")
    print(f"Mean Grade Score:     {mean_grade:.4f}")
    print("-" * 40)
    print("")

def generate_graphs():
    # 1. Find all CSV files
    csv_files = glob.glob(os.path.join(BENCHMARK_DIR, '*.csv'))
    
    # Sort files by version number so graph is chronological (v1, v2, v3...)
    csv_files.sort(key=get_version_number)

    if not csv_files:
        print(f"No CSV files found in {BENCHMARK_DIR}!")
        return

    results = []
    
    print(f"Found {len(csv_files)} benchmark files. Processing...\n")

    # 2. Process each CSV file
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # Create a nice name: "benchmark_v1.csv" -> "v1"
            version_name = os.path.basename(f).replace('.csv', '').replace('benchmark_', '')
            
            # Print the text report
            print_report_card(version_name, df)

            # Store stats for the graph
            results.append({
                'Version': version_name,
                'Precision': df['precision@10'].mean(),
                'Latency': df['duration'].mean()
            })
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

    df_results = pd.DataFrame(results)

    # --- GRAPH F: Engine Performance (Precision) ---
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['Version'], df_results['Precision'], marker='o', linestyle='-', color='#2c3e50', linewidth=2, markersize=8)
    
    plt.title('Engine Quality: Mean Precision@10 per Version', fontsize=14)
    plt.ylabel('Mean Precision@10', fontsize=12)
    plt.xlabel('Implementation Version', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.0, 1.0) # Fixed scale 0-1 for precision
    
    # Add value labels
    for i, v in enumerate(df_results['Precision']):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
        
    plt.savefig(os.path.join(OUTPUT_DIR, 'graph_f_performance.png'))
    print(f"Saved graph: {os.path.join(OUTPUT_DIR, 'graph_f_performance.png')}")
    plt.close()

    # --- GRAPH G: Retrieval Time (Latency) ---
    plt.figure(figsize=(10, 6))
    
    # Dynamic colors based on speed (<1.0s = Green, >1.0s = Gray/Red)
    colors = ['#27ae60' if l < 1.0 else '#95a5a6' for l in df_results['Latency']]
    
    bars = plt.bar(df_results['Version'], df_results['Latency'], color=colors)
    plt.title('Average Retrieval Time per Version', fontsize=14)
    plt.ylabel('Time (Seconds)', fontsize=12)
    plt.xlabel('Implementation Version', fontsize=12)
    
    # Target Line
    plt.axhline(y=1.0, color='r', linestyle='--', label='Target (<1.0s)')
    plt.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}s',
                 ha='center', va='bottom', fontweight='bold')

    plt.savefig(os.path.join(OUTPUT_DIR, 'graph_g_latency.png'))
    print(f"Saved graph: {os.path.join(OUTPUT_DIR, 'graph_g_latency.png')}")
    plt.close()

if __name__ == "__main__":
    generate_graphs()