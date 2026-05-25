import numpy as np
import pandas as pd
import json

# ==========================================
# 1. Configuration & Input Files
# ==========================================
# Your two specific inputs:
BISTRO_WORDS = "bistro_pairs.json"         # Generated from your triplets via Script 1
BISTRO_VECS = "bistro_vlinks_norm.npy"     # Generated from your triplets via Script 1

TARGET_VOCAB_FILE = "juridical_vocab.txt"  # new file: One word per line

# The PAISA infrastructure:
PAISA_PAIRS_CSV = "gold_pairs_50k.csv"     #800k ca
PAISA_VECS_50K = "Z_vecs_norm_50k.npy"
PAISA_WORDS_JSON = "Z_matrix_words.json"

#   OUTPUT:
OUTPUT_CSV = "targeted_analogies_results.csv"

TOP_N = 50

# MODE SWITCH
# False: "AND/OR" -> At least ONE word (C or D) must be in the vocab text file.
# True: "AND" -> BOTH words (C and D) must be in the vocab text file.
STRICT_MODE = False 

# ==========================================
# 2. Load Data & Extract Target Vocabulary
# ==========================================
print(f"Reading target vocabulary from {TARGET_VOCAB_FILE}...")
with open(TARGET_VOCAB_FILE, 'r', encoding='utf-8') as f:
    # .strip() removes whitespace/newlines, .lower() ensures matching
    target_vocab = {line.strip().lower() for line in f if line.strip()}
print(f"Loaded {len(target_vocab):,} unique target words.")

print("Loading Bistro and PAISA data...")
bistro_vlinks = np.load(BISTRO_VECS)
with open(BISTRO_WORDS, "r", encoding="utf-8") as f:
    bistro_pairs = json.load(f)
    
df_paisa = pd.read_csv(PAISA_PAIRS_CSV)
paisa_matrix_vecs = np.load(PAISA_VECS_50K, mmap_mode='r')
with open(PAISA_WORDS_JSON, "r", encoding="utf-8") as f:
    paisa_words = json.load(f)

# ==========================================
# 3. Filter the PAISA Pool
# ==========================================
print("Filtering PAISA pairs against the target vocabulary...")

# Map which PAISA matrix indices actually belong to your target vocabulary
valid_indices = {i for i, word in enumerate(paisa_words) if word in target_vocab}

mask_a = df_paisa['Idx_A'].isin(valid_indices)
mask_b = df_paisa['Idx_B'].isin(valid_indices)

if STRICT_MODE:
    final_mask = mask_a & mask_b
else:
    final_mask = mask_a | mask_b

df_filtered = df_paisa[final_mask].reset_index(drop=True)

print(f"Pairs reduced from {len(df_paisa):,} down to {len(df_filtered):,} valid targets.")

if len(df_filtered) == 0:
    raise ValueError("No PAISA pairs matched your target vocabulary. Halting.")

# ==========================================
# 4. Compute Filtered V-links in Memory
# ==========================================
print("Computing targeted PAISA v-links...")
idx_a = df_filtered['Idx_A'].values
idx_b = df_filtered['Idx_B'].values

raw_paisa_vlinks = paisa_matrix_vecs[idx_a] - paisa_matrix_vecs[idx_b]
norms = np.linalg.norm(raw_paisa_vlinks, axis=1, keepdims=True) + 1e-9
paisa_vlinks = raw_paisa_vlinks / norms

# ==========================================
# 5. Calculate Analogies
# ==========================================
print("Calculating similarity matrix...")
# Dot product: Bistro (N x 300) dot Filtered_PAISA_T (300 x M) -> Result (N x M)
sims = np.dot(bistro_vlinks, paisa_vlinks.T)

# ==========================================
# 6. Extract & Save Top Matches
# ==========================================
results = []
# Ensure we don't try to extract 50 if the filtered list is smaller than 50
actual_top_n = min(TOP_N, len(df_filtered))
print(f"Extracting top {actual_top_n} matches for each Bistro pair...")

for b_idx in range(len(bistro_pairs)):
    wA = bistro_pairs[b_idx]['Word_A']
    wB = bistro_pairs[b_idx]['Word_B']
    relation = bistro_pairs[b_idx].get('Relation', 'unknown')
    
    row_sims = sims[b_idx]
    
    # Grab the highest scoring indices
    top_paisa_indices = np.argsort(row_sims)[-actual_top_n:][::-1]
    
    for rank, p_idx in enumerate(top_paisa_indices):
        score = float(row_sims[p_idx])
        
        # Look up the PAISA words using the FILTERED dataframe
        idC = int(df_filtered.iloc[p_idx]['Idx_A'])
        idD = int(df_filtered.iloc[p_idx]['Idx_B'])
        wC, wD = paisa_words[idC], paisa_words[idD]
        
        results.append({
            'Bistro_A': wA,
            'Bistro_B': wB,
            'Relation': relation,
            'Paisa_C': wC,
            'Paisa_D': wD,
            'Score': score,
            'Rank': rank + 1
        })

df_results = pd.DataFrame(results)
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"Done! Saved {len(df_results):,} analogies to {OUTPUT_CSV}.")