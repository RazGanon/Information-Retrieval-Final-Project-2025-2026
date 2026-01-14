import json
import requests
import time
import pandas as pd
import numpy as np

# CONFIGURATION
# ---------------------------------------------------------
# CHANGE THIS to your external GCP IP if running from laptop
SERVER_URL = 'http://34.172.210.231:8080/' 
QUERIES_FILE = 'queries_train.json'
OUTPUT_FILE = 'benchmark_v2.csv'
# ---------------------------------------------------------

def get_metrics(true_ids, pred_ids, k):
    """Calculate Precision@K, Recall@K, and F1@K"""
    true_set = frozenset(true_ids)
    pred_list = pred_ids[:k]
    
    if len(pred_list) == 0:
        precision = 0.0
    else:
        precision = len([doc for doc in pred_list if doc in true_set]) / len(pred_list)
        
    if len(true_set) == 0:
        recall = 0.0
    else:
        recall = len([doc for doc in pred_list if doc in true_set]) / len(true_set)
        
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        
    return precision, recall, f1

def get_map(true_ids, pred_ids):
    """Calculate Mean Average Precision"""
    true_set = frozenset(true_ids)
    precisions = []
    hits = 0
    for i, doc in enumerate(pred_ids):
        if doc in true_set:
            hits += 1
            precisions.append(hits / (i + 1))
    if len(true_set) == 0: return 0.0
    return sum(precisions) / len(true_set)

# 1. LOAD QUERIES
with open(QUERIES_FILE, 'rt') as f:
    queries = json.load(f)

results = []
print(f"Testing {len(queries)} queries against {SERVER_URL}...")
print("---------------------------------------------------------")

# 2. RUN BENCHMARK
for query, true_wids in queries.items():
    true_wids = [str(x) for x in true_wids]
    
    start_time = time.time()
    try:
        # Use the main search endpoint
        res = requests.get(f"{SERVER_URL}/search", params={'query': query}, timeout=35)
        duration = time.time() - start_time
        
        if res.status_code == 200:
            pred_wids = [str(doc[0]) for doc in res.json()]
            
            # --- CALCULATE METRICS FOR FULL REQUIREMENTS ---
            
            # 1. Precision@5 (Part 1 of Quality Score)
            p_5, _, _ = get_metrics(true_wids, pred_wids, k=5)
            
            # 2. F1@30 (Part 2 of Quality Score)
            _, _, f1_30 = get_metrics(true_wids, pred_wids, k=30)
            
            # 3. Precision@10 (For Minimum Requirement)
            p_10, _, _ = get_metrics(true_wids, pred_wids, k=10)
            
            # 4. MAP (General Quality)
            map_score = get_map(true_wids, pred_wids)
            
            # 5. THE "GRADE SCORE" (Harmonic Mean of P@5 and F1@30)
            if p_5 + f1_30 == 0:
                grade_score = 0.0
            else:
                grade_score = (2 * p_5 * f1_30) / (p_5 + f1_30)

            results.append({
                'query': query,
                'duration': duration,
                'precision@5': p_5,
                'precision@10': p_10,
                'f1@30': f1_30,
                'grade_score': grade_score, # This is the "Harmonic Mean" for Req #3
                'map': map_score
            })
            
            # Print brief status
            print(f"Query: {query[:20]}... | Time: {duration:.2f}s | Score: {grade_score:.3f}")
            
        else:
            print(f"Error {res.status_code} for query: {query}")
            
    except Exception as e:
        print(f"Exception for query {query}: {e}")

# 3. AGGREGATE RESULTS
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)

# 4. FINAL REPORT CARD
print("\n" + "="*40)
print("       BENCHMARK REPORT CARD")
print("="*40)
if not df.empty:
    print(f"Queries Processed:    {len(df)}")
    print(f"Average Latency:      {df['duration'].mean():.4f}s  (Target: < 1.0s)")
    print(f"Mean Precision@10:    {df['precision@10'].mean():.4f}  (Pass: > 0.1)")
    print(f"Mean Grade Score:     {df['grade_score'].mean():.4f}  (Harmonic Mean P@5 & F1@30)")
    print("-" * 40)
    print(f"Data saved to {OUTPUT_FILE}")
else:
    print("No results returned. Check server connection.")