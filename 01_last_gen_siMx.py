import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import os
import json
from data_utils import extra_get_common_words_with_pos

# -------------------- CONFIGURATION ------------------------
conll_file = r"c:\Users\Utente\Desktop\VSCODE\Paisa\paisa.annotated.CoNLL.utf8"
target_tags = ['S', 'V', 'A']
EXTRA_WORDS_FILE = 'prova_extra.txt'

# Load the extra words into a list
with open(EXTRA_WORDS_FILE, 'r', encoding='utf-8') as f:
    bistro_word_vectors = [line.strip() for line in f if line.strip()]

# -------------------- DATA EXTRACTION ------------------------
print("Processing CoNLL file ...")
most_common, word_to_pos = extra_get_common_words_with_pos(
    conll_file, target_tags, top_n=50000, extra_words_file=EXTRA_WORDS_FILE
)

with open("top_words.json", 'w', encoding='utf-8') as f:
    json.dump(most_common, f, indent=4)

# Load Model
model = Word2Vec.load('analogy_m.model')

# -------------------- SAFE VOCABULARY SLICING ----------------
# We must ensure the extra words are prioritized and not cut off by a hard slice
extra_words_set = set(bistro_word_vectors)
valid_top_words = []

# Pass 1: Guarantee the extra words (if present in the model) get added first
for w, freq in most_common:
    if w in extra_words_set and w in model.wv:
        valid_top_words.append(w)

# Pass 2: Fill the rest of the list up to exactly 50,000 using standard PAISA words
for w, freq in most_common:
    if w not in extra_words_set and w in model.wv:
        if len(valid_top_words) < 50000:
            valid_top_words.append(w)

print(f"Final vocabulary size: {len(valid_top_words)} words.")

with open("Z_matrix_words.json", "w", encoding="utf-8") as f:
    json.dump(valid_top_words, f)

word_to_pos_filtered = {w: word_to_pos[w] for w in valid_top_words if w in word_to_pos}
with open("word_to_pos.json", "w", encoding="utf-8") as f:
    json.dump(word_to_pos_filtered, f) 

# ------------------- EXTRACT AND NORMALIZE -----------------
print(f"Extracting vectors for {len(valid_top_words)} words...")
word_vectors = model.wv[valid_top_words].astype('float32') 

length = np.linalg.norm(word_vectors, axis=1, keepdims=True) 
vecs_norm = word_vectors / (length + 1e-9) 

# ------------------- SIMILARITY MATRIX -----------------
print("Calculating Square Similarity Matrix ...")
# Creating a square 50k x 50k matrix so the row/col indexing works perfectly
similarity_matrix = np.dot(vecs_norm, vecs_norm.T) 

print(f"Matrix Complete! Shape: {similarity_matrix.shape}")
print(f"Memory Usage: {similarity_matrix.nbytes / (1024**3):.2f} GB")

print("Saving matrix to disk...")
np.save("Z_similarity_matrix_50k.npy", similarity_matrix)
np.save("Z_vecs_norm_50k.npy", vecs_norm)

print("Done! Files saved: Z_similarity_matrix_50k.npy, Z_vecs_norm_50k.npy and Z_matrix_words.json")