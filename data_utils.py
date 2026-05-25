import os
from gensim.models.callbacks import CallbackAny2Vec
from collections import defaultdict, Counter
from spellchecker import SpellChecker
import numpy as np
import pandas as pd

# -----------------------1) MEMORY EFFICIENT ITERATOR ----------------------
# A memory efficient iterator that yields sentences as lists of words
# keeps in RAM just one sentence at the time (current_sentence) and yields it 

# a) define a class that implements __iter__
# b) define the iterator
# c) tokenize and clean every sentence of the database
class SentenceIterator: 
    def __init__(self, filepath, use_lemma=True):
        self.filepath = filepath
        self.use_lemma = use_lemma  # Store the choice imput from train
    
    def __iter__(self):
        print(f"Reading {self.filepath} (Mode: {'Lemma' if self.use_lemma else 'Raw Word'})...")
        with open(self.filepath, 'r', encoding='utf-8') as f:
            current_sentence = []
            for line in f:
                line = line.strip()
                if line.startswith('#'): continue
                if not line:                      # EMPTY LINE MARKS END OF SENTENCE 
                    if current_sentence:
                        yield current_sentence
                        current_sentence = []
                    continue
                
                parts = line.split('\t')
                if len(parts) < 4: continue
                
                pos_tag = parts[3]
                if pos_tag == "PUNCT": continue

                # THE SWITCH - imput from the call to the function 
                
                if self.use_lemma:
                    token = parts[2] 
                else:
                    token = parts[1]

                current_sentence.append(token.lower())
            
            if current_sentence:
                yield current_sentence      # YIELD THE LAST SENTENCE IF PRESENT 





#---------------------- 2) IS REAL WORD ------------------------

# a) drop single letters
# b) lowercase
# c) alphabetic, drops symbols,...
# d) spellcheck, are the words italian words?

spell = SpellChecker(language='it')

def is_real_word(lemma):
    

    word_lower = lemma.lower()

    if len(word_lower) < 2:
        return False
    
    if not word_lower.isalpha(): 
        return False
    
    if word_lower not in spell:
        return False
    
    return True

# ---------------------- 3) SORT BY FREQUENCY (+POS)  -----------------------------------

# a) fuction that get as argouments the database, the tags of interest and the top_words number
# b) for cycle that count the number of occurences of each word and its tag
#

def get_common_words_with_pos(filepath, target_tags=['S', 'V', 'A'], top_n=50000):
    print(f"Counting occurrences and POS tags in {filepath}...")
    
    # initialize defaultdict, to avoid getting error if the key doesnt exist
    # the value of the non-existing key will be 0 =int()
    # if counts= {} then counts[(lemma, tag)] += 1 will raise a key error: 'roma' has no number to add up to 

    counts = defaultdict(int)

    #EXAMPLE OF DIFFERENT DICTIONARIES: 
    # 1) vanilla: just stores {"book": N}
    # 2) auto_zero: stores id 0 if key doesnt exist {"book": 0}
    # 3) tally_counter: it has a built-in most_common method

    lemma_total_freq = Counter() # freq of each lemma (indipendently of the tag)
    dominant_pos = {} # higest number of times a tag is found 
    winning_tag_score = defaultdict(int) # stores the winning string ex. 'N'
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:

            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if len(parts) < 4: continue
            
            lemma, tag = parts[2], parts[3]
            
            if tag in target_tags and is_real_word(lemma):
                counts[(lemma, tag)] += 1
    
    print("Aggregating frequencies...")

    # KEY= (lemma,tag)
    # VALUE= freq

    for (lemma, tag), freq in counts.items():
        lemma_total_freq[lemma] += freq #'book' = 100 
        if freq > winning_tag_score[lemma]: # if 100> 0 
            winning_tag_score[lemma] = freq # winning_tag_score['book'] = 100
            dominant_pos[lemma] = tag # dominant_pos['book'] = 'N'
            
    top_words = lemma_total_freq.most_common(top_n)
    word_to_pos = {word: dominant_pos[word] for word, _ in top_words} # get the word from top_words and find the tag in dominant_pos
    
    print(f"Success: Processed {len(word_to_pos)} words.")
    return top_words, word_to_pos



"""
    Args:
        matrix: The mmap numpy array.
        num_words: Total number of words (matrix.shape[0]).
        chunk_size: Number of rows to process at once.
        k: The number of top neighbors to consider.
        min_sim: Optional absolute floor to drop completely unrelated pairs.
    """

def get_mnn(matrix, num_words, chunk_size=1000, k=20, min_sim=0.0):
    
    top_k_sets = []
    
    print(f"Phase 1: Finding top {k} neighbors for each word...")
    for start_row in range(0, num_words, chunk_size):
        end_row = min(start_row + chunk_size, num_words)
        chunk = matrix[start_row:end_row, :]
        
        # np.argpartition is significantly faster and more memory efficient than np.argsort.
        # We grab k+1 because the word's similarity to itself is 1.0 (always a top match).
        top_indices = np.argpartition(chunk, -(k+1), axis=1)[:, -(k+1):]
        
        # Convert to sets for O(1) lookups in Phase 2
        for i, row_indices in enumerate(top_indices):
            global_idx = start_row + i
            neighbors = set(row_indices)
            # Discard the word matching with itself
            neighbors.discard(global_idx) 
            top_k_sets.append(neighbors)
            
        progress = (end_row / num_words) * 100
        print(f"  Scanning chunks: {progress:.1f}%", end='\r')
        
    print("\nPhase 2: Identifying mutual relationships...")
    results = []
    
    # We only check the upper triangle (j > i) to prevent duplicate A-B, B-A entries.
    for i in range(num_words):
        for j in top_k_sets[i]:
            if j > i and i in top_k_sets[j]:
                score = float(matrix[i, j])
                
                # Optional safety net: drop pairs if similarity is extremely low
                if score >= min_sim:
                    results.append({
                        'Idx_A': i,
                        'Idx_B': int(j),
                        'Score': score
                    })
                    
    return results



"""
    Extracts pairs that match the geometric trajectory of provided seed pairs.
    
    Args:
        seeds_df: DataFrame containing 'Word_A' and 'Word_B' columns.
        vecs: The mmap numpy array of normalized word vectors (e.g., 50k x 300).
        words: List of the 50k vocabulary words.
        chunk_size: Number of rows to process at once.
        min_score: Minimum cosine similarity between the predicted vector and the found word.
    """

def get_seed_offsets(seeds_df, vecs, words, chunk_size=1000, min_score=0.35):
    
    # 1. Fast lookup dictionary for words to their row index
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    
    # 2. Extract valid seed vectors
    valid_offsets = []
    
    print("Validating seed pairs against vocabulary...")
    for _, row in seeds_df.iterrows():
        word_a, word_b = row['Word_A'], row['Word_B']
        
        if word_a in word_to_idx and word_b in word_to_idx:
            idx_a = word_to_idx[word_a]
            idx_b = word_to_idx[word_b]
            
            # The relationship vector: A -> B
            offset = vecs[idx_a] - vecs[idx_b]
            valid_offsets.append(offset)
        else:
            print(f"  [Skipped] Seed pair not in vocab: {word_a} - {word_b}")
            
    if not valid_offsets:
        raise ValueError("None of the seed pairs were found in the 50k vocabulary.")
        
    # 3. Calculate the average relationship trajectory
    avg_offset = np.mean(valid_offsets, axis=0)
    
    num_words = vecs.shape[0]
    results = []
    
    print("\nApplying vector trajectory across the 50k vocabulary...")
    # 4. Process the vocabulary in chunks to prevent RAM overflow
    for start_row in range(0, num_words, chunk_size):
        end_row = min(start_row + chunk_size, num_words)
        
        # Grab a chunk of vectors (Word C)
        chunk_vecs = vecs[start_row:end_row, :]
        
        # Calculate the predicted target (Word D)
        # Standard analogy logic: D = C - (A - B)
        target_vecs = chunk_vecs - avg_offset
        
        # L2 Normalize the target vectors so dot product = cosine similarity
        norms = np.linalg.norm(target_vecs, axis=1, keepdims=True)
        # Avoid division by zero just in case
        target_vecs_norm = np.divide(target_vecs, norms, out=np.zeros_like(target_vecs), where=norms!=0)
        
        # Matrix multiplication to find similarities across the entire 50k space
        # target_vecs_norm (1000 x 300) dot vecs.T (300 x 50000) -> sims (1000 x 50000)
        sims = np.dot(target_vecs_norm, vecs.T)
        
        # 5. Extract the best match for each word in the chunk
        for i, row_sims in enumerate(sims):
            global_idx_c = start_row + i
            
            # We don't want the word to match with itself
            row_sims[global_idx_c] = -1.0 
            
            # Find the index of the highest similarity
            best_match_idx = np.argmax(row_sims)
            best_score = float(row_sims[best_match_idx])
            
            # Keep it if it meets our structural threshold
            if best_score >= min_score:
                results.append({
                    'Idx_A': global_idx_c,
                    'Idx_B': int(best_match_idx),
                    'Score': best_score
                })
                
        progress = (end_row / num_words) * 100
        print(f"  Scanning chunks: {progress:.1f}%", end='\r')
        
    print("\nDone extracting trajectory pairs.")
    return results




def switch_get_seed_offsets(seeds_df, vecs, words, chunk_size=1000, min_score=0.35, mode='mean'):
    """
    Extracts pairs based on geometric trajectories from seed pairs.
    
    Args:
        seeds_df: DataFrame containing 'Word_A', 'Word_B', and 'Relation' columns.
        vecs: The mmap numpy array of normalized word vectors.
        words: List of the 50k vocabulary words.
        chunk_size: Number of rows to process at once.
        min_score: Minimum cosine similarity.
        mode: 'mean' averages the vectors for each 'Relation'.
              'individual' tests the exact offset of every single pair independently.
    """
    word_to_idx = {word: idx for idx, word in enumerate(words)}
    
    # 1. Group and calculate offsets based on the chosen mode
    offsets_dict = {}  # Format: {label: offset_vector}
    
    print(f"Preparing offsets in '{mode}' mode...")
    
    if mode == 'mean':
        # Group pairs by their 'Relation' column and calculate the mean offset
        for relation, group in seeds_df.groupby('Relation'):
            valid_offsets = []
            for _, row in group.iterrows():
                wa, wb = row['Word_A'], row['Word_B']
                if wa in word_to_idx and wb in word_to_idx:
                    valid_offsets.append(vecs[word_to_idx[wa]] - vecs[word_to_idx[wb]])
            
            if valid_offsets:
                offsets_dict[relation] = np.mean(valid_offsets, axis=0)
                print(f"  Mapped Relation: '{relation}' (averaged {len(valid_offsets)} pairs)")
                
    elif mode == 'single':
        # Calculate the exact offset for every single pair independently
        for _, row in seeds_df.iterrows():
            wa, wb = row['Word_A'], row['Word_B']
            if wa in word_to_idx and wb in word_to_idx:
                label = f"{wa}_{wb}"  # Use the actual words as the label
                offsets_dict[label] = vecs[word_to_idx[wa]] - vecs[word_to_idx[wb]]
                
    if not offsets_dict:
        raise ValueError("No valid seed pairs found in the vocabulary.")

    num_words = vecs.shape[0]
    results = []
    
    print(f"\nApplying {len(offsets_dict)} unique trajectory(s) across the vocabulary...")
    
    # 2. Process the vocabulary in chunks to prevent RAM overflow
    for start_row in range(0, num_words, chunk_size):
        end_row = min(start_row + chunk_size, num_words)
        chunk_vecs = vecs[start_row:end_row, :]
        
        # 3. Apply EACH offset trajectory to the current chunk of words
        for label, offset in offsets_dict.items():
            
            target_vecs = chunk_vecs - offset #get the ghost vector (word D)
            
            norms = np.linalg.norm(target_vecs, axis=1, keepdims=True)
            target_vecs_norm = np.divide(target_vecs, norms, out=np.zeros_like(target_vecs), where=norms!=0)
            
            sims = np.dot(target_vecs_norm, vecs.T) #check wich real_vectors are the closer one to ghost_word D. 
            
            # 4. Extract the best match
            for i, row_sims in enumerate(sims):
                global_idx_c = start_row + i
                row_sims[global_idx_c] = -1.0 
                
                best_match_idx = np.argmax(row_sims)
                best_score = float(row_sims[best_match_idx])
                
                if best_score >= min_score:
                    results.append({
                        'Offset_Source': label,  # Tracks which relation/pair found this result
                        'Idx_A': global_idx_c,
                        'Idx_B': int(best_match_idx),
                        'Score': best_score
                    })
                    
        progress = (end_row / num_words) * 100
        print(f"  Scanning chunks: {progress:.1f}%", end='\r')
        
    print("\nDone extracting trajectory pairs.")
    return results



#WITH LIO:

from collections import defaultdict, Counter
import os # Assuming you might need this for file paths

# Make sure is_real_word is defined somewhere above this function
# def is_real_word(lemma): ...

def extra_get_common_words_with_pos(filepath, target_tags=['S', 'V', 'A'], top_n=50, extra_words_file= "prova_extra.txt"):
    print(f"Counting occurrences and POS tags in {filepath}...")
    
    # initialize defaultdict, to avoid getting error if the key doesnt exist
    counts = defaultdict(int)

    lemma_total_freq = Counter() # freq of each lemma (independently of the tag)
    dominant_pos = {} # highest number of times a tag is found 
    winning_tag_score = defaultdict(int) # stores the winning string ex. 'N'

    count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
           # count += 1                                       #da togliere
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if len(parts) < 4: continue
            
            lemma, tag = parts[2], parts[3]
            
            if tag in target_tags and is_real_word(lemma):
                counts[(lemma, tag)] += 1

            if count==10000:
                break
    
    print("Aggregating frequencies...")

    # KEY= (lemma,tag)
    # VALUE= freq
    for (lemma, tag), freq in counts.items():
        lemma_total_freq[lemma] += freq 
        if freq > winning_tag_score[lemma]: 
            winning_tag_score[lemma] = freq 
            dominant_pos[lemma] = tag 
            
    top_words = lemma_total_freq.most_common(top_n)
    word_to_pos = {word: dominant_pos[word] for word, _ in top_words} 
    
    # --- NEW FEATURE: Add extra words from optional txt file ---
    if extra_words_file:
        print(f"Adding extra words from {extra_words_file}...")
        # Use a set for O(1) fast lookups to see if we already added the word
        #.keys extract just the words
        existing_words = set(word_to_pos.keys())
        
        with open(extra_words_file, 'r', encoding='utf-8') as f_extra:
            for line in f_extra:
                word = line.strip() # Remove \n and whitespace
                
                # Skip empty lines or words already in our top 50k
                if not word or word in existing_words:
                    continue
                
                # If the word was in the corpus but missed the top_n cutoff, grab its real data.
                # If it's completely unseen, default to freq=0 and POS='UNK'.
                freq = lemma_total_freq.get(word, 0)
                pos = dominant_pos.get(word, 'UNK') 
                
                top_words.append((word, freq))
                word_to_pos[word] = pos
                existing_words.add(word)

    print(f"Success: Processed {len(word_to_pos)} words.")
    return top_words, word_to_pos







# OLD: JUST COMMON WORDS


def get_common_words(filepath, target_tags=['S', 'V', 'A'] , top_n=20000):
    print(f"Counting occurrences in {filepath}...")
    word_freq = Counter()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split('\t')   
            if len(parts) < 4:
                continue
            
            lemma = parts[2]
            tag = parts[3]
            
            if tag in target_tags:

                if is_real_word(lemma): 
                    word_freq[lemma] += 1
                
    return word_freq.most_common(top_n)






