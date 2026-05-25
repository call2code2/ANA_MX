import numpy as np
import pandas as pd
import json

# 1. Configuration
SIM_MATRIX_PATH = "Z_similarity_matrix_50k.npy"
WORDS_PATH = "Z_matrix_words.json"
CHUNK_SIZE = 1000  
LOWER_LIMIT = 0.35
UPPER_LIMIT = 1.0

print(f"Loading matrix {SIM_MATRIX_PATH}...")

with open(WORDS_PATH, encoding='utf-8') as f:
    words = json.load(f)

matrix = np.load(SIM_MATRIX_PATH, mmap_mode='r')
num_words = matrix.shape[0]

results = []

print(f"Extracting pairs in range {LOWER_LIMIT} - {UPPER_LIMIT}...")

# 2. Loop through the matrix in chunks
for start_row in range(0, num_words, CHUNK_SIZE):
    end_row = min(start_row + CHUNK_SIZE, num_words)
    
    chunk = matrix[start_row:end_row, :]
    
    rows, cols = np.where((chunk >= LOWER_LIMIT) & (chunk <= UPPER_LIMIT))
    
    global_rows = rows + start_row
    
    mask = cols > global_rows
    final_rows = global_rows[mask]
    
    final_cols = cols[mask]
    final_scores = chunk[rows[mask], final_cols]

    for i in range(len(final_rows)):
        # Cast to int to ensure safe list indexing
        idx_a = int(final_rows[i])
        idx_b = int(final_cols[i])
        
        results.append({
            'Idx_A': idx_a,
            'Idx_B': idx_b,
            'Word_A': words[idx_a],
            'Word_B': words[idx_b],
            'Score': float(final_scores[i])
        })
    
    progress = (end_row / num_words) * 100
    # Overwrite the print line for a cleaner terminal output
    print(f"Progress: {progress:.1f}% | Found so far: {len(results):,}", end='\r')

print("\nSaving to CSV...") # Newline after the progress loop

# 3. Save to CSV
df = pd.DataFrame(results)
df.to_csv("gold_pairs_50k.csv", index=False)
print(f"Success! Saved {len(results):,} pairs to gold_pairs_50k.csv")