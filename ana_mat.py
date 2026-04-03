import numpy as np
import pandas as pd
import json
import heapq
from spellchecker import SpellChecker


# SECTIONS:
#  1) LOAD DATA
#  2) GENERATE V-LINKS
#  3) ANALOGY MATRIX (DOUBLE BATCHING)
#    a) np.dot 1kx50k(for every batch)
#    b) buoncer: find the best tile candidates
#    c) translation loop: from local idx to matrix words
#    d) vector_extraction + filters
#    e) push to the heap
#  4) SAVE


#==================================================================================================================================
#==================================================================================================================================


# ------------- 1. LOAD PRE-CALCULATED DATA ---------------

print("Loading word vectors and Golden Standard...")


# Load the "Legend" and "Coordinates" 
with open("Z_matrix_words.json", "r", encoding="utf-8") as f: #all the matrix words
    words = json.load(f)


# Load the POS data from the disk instead of recalculating it
print("Loading POS mapping from JSON...")
with open("word_to_pos.json", "r", encoding="utf-8") as f:
    word_to_pos = json.load(f)

vecs_norm = np.load("Z_vecs_norm_50k.npy") # coordinates
df_golden = pd.read_csv("gold_pairs_50k.csv", sep=',')  


num_pairs = len(df_golden)

months= {"gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"}

with open("geo_stopwords.txt", "r", encoding="utf-8") as f:
    geo_words = {line.strip().lower() for line in f}


#==================================================================================================================================
#==================================================================================================================================


#  ------------------ 2. GENERATE V-LINKS (MEMORY EFFICIENT) -----------------

print("Generating relationship vectors (V-links) in chunks...")

v_links_norm = np.zeros((num_pairs, 300), dtype='float32') # Pre-allocate memory

# Process in chunks to save RAM
chunk_size_gen = 100000
for start in range(0, num_pairs, chunk_size_gen):
    end = min(start + chunk_size_gen, num_pairs)
    chunk = df_golden.iloc[start:end]
    
# NEW VERSION (USES INDICES)
    for i, (_, row) in enumerate(chunk.iterrows()):
        idx_a = int(row['Idx_A']) # Use the number from the CSV
        idx_b = int(row['Idx_B'])

        v_a = vecs_norm[idx_a]
        v_b = vecs_norm[idx_b]
        

        v_link = v_a - v_b

        # Normalize immediately
        norm = np.linalg.norm(v_link) + 1e-9
        v_links_norm[start + i] = v_link / norm

print(f"Matrix created: {v_links_norm.nbytes / 1e9:.2f} GB in RAM.")



# =====================================================================
# 3. ANALOGY SEARCH (TILED DOUBLE-BATCHING)
# =====================================================================


# DOUBLE TILED BATCHING
# 1) OUTER LOOP: loop through 1000 queries (row)
#    r_start= the starting index of the rows
#    r_end= boundary of the current row
#       start_crate= the starting index of the columns // skpis the queries already done every 50k

# 2) INNER LOOP: for each R loop through 50k (column)
#    c_start= the starting index of the columns
#    c_end= boundary of the current column
#  
# 3) NP.DOT: calculates 1kx50k analogy matrix
#
# 4) BOUNCER: find the best tile candidates
#    argpartition: returns the indices

# 5) TRANSLATION LOOP:
#    outer loop: translates the local row (0 to 999) into the true df_golden idx (r_global).  Base Pair (A:B).
#    inner loop: for every row, translates the 100 winning local columns into their true df_golden idx (c_global). Candidate Pair (C:D).

# 6) load into pairs1,2 the rows of the df_golden (A:B) and (C:D)

# 7) get the original matrix index for pair1(wordA,B) and pair2(wordC,D)

# 8) using the id(A,B,C,D)get the words from the matrix words


top_n = 1000
analogy_heap = []
count = 0
word_counts = {}
max_limit = 10000

# These sizes keep your RAM usage around 200MB per calculation
batch_size_r = 1000   # How many "Query" pairs we look at at once - ROWS
chunk_size_c = 50000  # How many "Database" pairs we compare them against - COLUMNS

print(f"Starting Tiled Search on {num_pairs} pairs...")




#1)
for r_start in range(0, num_pairs, batch_size_r):
    r_end = min(r_start + batch_size_r, num_pairs)
    
    start_crate = (r_start // chunk_size_c) * chunk_size_c

#2)    
    for c_start in range(start_crate, num_pairs, chunk_size_c):
        c_end = min(c_start + chunk_size_c, num_pairs)
        
#3)
        scores = np.dot(v_links_norm[r_start:r_end], v_links_norm[c_start:c_end].T) #50mln grid
#4)      
        num_c = min(100, scores.shape[1]) 
        candidate_idx = np.argpartition(scores, -num_c, axis=1)[:, -num_c:] 
        
#5)        
        for r_local in range(len(candidate_idx)): #1k rloc idx (0:1001)
            r_global = r_start + r_local # Actual row
            
            for c_local in candidate_idx[r_local]: 
                c_global = c_start + c_local # Actual column
                
                # Rule: Avoid self-comparison and mirror duplicates
                if r_global >= c_global: continue 
                
                raw_score = scores[r_local, c_local] # similarity score of the 2 pairs
                
                # Check absolute value (catches both 0.50 and -0.50)
                if abs(raw_score) < 0.40: 
                    continue
#6)
                
                pair_1 = df_golden.iloc[r_global]
                pair_2 = df_golden.iloc[c_global]
#7)
                idA, idB = int(pair_1['Idx_A']), int(pair_1['Idx_B'])
                idC, idD = int(pair_2['Idx_A']), int(pair_2['Idx_B'])
#8)
                # Look up the actual words
                wA, wB = words[idA], words[idB]
                wC, wD = words[idC], words[idD]

                # THE AUTO-FLIPPER
                if raw_score < 0:
                    wC, wD = wD, wC     # Swap C and D!
                    score = abs(raw_score) # Save as positive for the heap
                else:
                    score = raw_score

                if len(analogy_heap) == top_n and score <= analogy_heap[0][0]:
                    continue
                
                

                #------------- FILTERS AND VECTOR EXTRACTION ---------------

                word_set = {wA, wB, wC, wD}
                
                if len(word_set) == 4:


                # MONTHS AND GEOWORDS CHECK
                    
                    if any(w in months for w in word_set) :continue

                    if any(w.lower() in geo_words for w in [wA, wB, wC, wD]):continue
                        
                # VECTOR EXTRACTION    
                    v_a, v_b, v_c, v_d = vecs_norm[idA], vecs_norm[idB], vecs_norm[idC], vecs_norm[idD]


                # CLOSE SYNONYMS CHECKS
                    dist_ab, dist_ac = np.linalg.norm(v_a - v_b), np.linalg.norm(v_a - v_c)
                    dist_bd, dist_cd = np.linalg.norm(v_b - v_d), np.linalg.norm(v_c - v_d)
                    
                    if np.dot(v_a,v_c) > 0.6: continue
                    if np.dot(v_b,v_d) > 0.6: continue

                # COUNT FOR FILTERING
                    if any(word_counts.get(w, 0) >= max_limit for w in word_set): continue


                    analogy_data = {
                        'Word_A': wA, 'Word_B': wB, 'Word_C': wC, 'Word_D': wD,
                        'Score': float(score), 
                        #'Solver_Score': float(solver_score),
                        #'Attr_Sim': float(avg_attr_sim)
                    }

                    if len(analogy_heap) < top_n:
                        heapq.heappush(analogy_heap, (score, count, analogy_data))
                    else:
                        heapq.heappushpop(analogy_heap, (score, count, analogy_data))
                    
                    #word_counts= keeps track of the word usage
                    #.get(w,0)= checks the word 8if not appeared starts at 0)
                    for w in word_set:
                        word_counts[w] = word_counts.get(w, 0) + 1
                    count += 1 #tie breaker

    # Heartbeat: Print progress after every 1000 rows are fully checked against all 6M
    print(f"Progress: {r_end}/{num_pairs} pairs fully searched.")

# =====================================================================
# =====================================================================

# 4. SAVE RESULTS
if not analogy_heap:
    print("No analogies found.")
else:
    final_list = [item[2] for item in analogy_heap]
    df_analogies = pd.DataFrame(final_list).sort_values(by='Score', ascending=False)
    df_analogies.to_csv("top_1000_sharpest_analogies.csv", index=False, sep=';')
    print("Done! Top 1,000 analogies saved to 'top_1000_sharpest_analogies.csv'.")