import numpy as np
import json
import re
from gensim.models import Word2Vec

# 1. Configuration
#   INPUTS:
MODEL_PATH = "analogy_m.model"       
BISTRO_TXT = "prova_triplets.txt"

#   OUTPUTS:
OUTPUT_VECS = "bistro_vlinks_norm.npy"
OUTPUT_WORDS = "bistro_pairs.json"
MISSING_FILE = "b_missing_w.txt"

print(f"Loading Word2Vec model from {MODEL_PATH}...")
model = Word2Vec.load(MODEL_PATH)

valid_pairs = []
v_links = []
missing_words = set()

print("Processing Bistro pairs...")
# 2. Read TXT and Extract Vectors
with open(BISTRO_TXT, "r", encoding="utf-8") as f:
    for line in f:
        clean_line = line.strip()
        if not clean_line:
            continue
            
        # Regex split: cuts the line using Tab (\t), Pipe (|), or Slash (/)
        parts = re.split(r'[\t|/\\]+', clean_line)      
          
        # Ensure we have at least 2 parts; grab the 3rd if it exists
        if len(parts) >= 3:
            wA, wB, relation = parts[0].strip(), parts[1].strip(), parts[2].strip()
        elif len(parts) == 2:
            wA, wB, relation = parts[0].strip(), parts[1].strip(), "unknown"
        else:
            continue # Skip malformed lines
            
        # 3. Check for missing words
        missing_a = wA not in model.wv
        missing_b = wB not in model.wv
        
        if missing_a: missing_words.add(wA)
        if missing_b: missing_words.add(wB)
            
        if missing_a or missing_b:
            continue
            
        # 4. Extract raw vectors
        v_a_raw = model.wv[wA]
        v_b_raw = model.wv[wB]
        
        # Step A: Normalize the individual words to remove frequency bias
        v_a = v_a_raw / (np.linalg.norm(v_a_raw) + 1e-9)
        v_b = v_b_raw / (np.linalg.norm(v_b_raw) + 1e-9)
        
        # Step B: Compute the pure semantic relationship
        v_link = v_a - v_b
        
        # Step C: Normalize the resulting v-link for Cosine Similarity scaling
        v_link_norm = v_link / (np.linalg.norm(v_link) + 1e-9)
        
        # Save the Relation alongside the words!
        valid_pairs.append({'Word_A': wA, 'Word_B': wB, 'Relation': relation})
        v_links.append(v_link_norm)

# 5. Save Outputs
if missing_words:
    with open(MISSING_FILE, "w", encoding="utf-8") as f:
        for w in sorted(missing_words):
            f.write(f"{w}\n")
    print(f"Saved {len(missing_words)} missing words to {MISSING_FILE}.")

np.save(OUTPUT_VECS, np.array(v_links, dtype='float32'))
with open(OUTPUT_WORDS, "w", encoding="utf-8") as f:
    json.dump(valid_pairs, f)
    
print(f"Success! Extracted {len(valid_pairs)} normalized Bistro v-links.")