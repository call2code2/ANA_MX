import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import os
import json
from data_utils import get_common_words_with_pos


# SECTIONS:
# 1) LOAD DATA: most_common, word_to_pos
# 2) STORE DATA: word_to_pos, matrix words
# 3) EXTRACT WV and NORMALIZE
# 4) GENERATE THE MATRIX 
# 5) SAVE



conll_file = r"c:\Users\Utente\Desktop\VSCODE\Paisa\paisa.annotated.CoNLL.utf8"
target_tags = ['S', 'V', 'A' ]


# --------------------  DATA EXTRACTION ------------------------

# 1. GET COMMON WORDS WITH POS 

print("Processing CoNLL file ...")

most_common, word_to_pos = get_common_words_with_pos(conll_file, target_tags, top_n=50000)

# 2. STORE most_common TO topW_file 
with open("top_words.json", 'w', encoding='utf-8') as f:
    json.dump(most_common, f, indent=4)
    

# ------------------- SIMILARITY MATRIX -----------------

# 1. LOAD THE MODEL
model=Word2Vec.load('analogy_m.model')


#2. SLICE and STORE TOP WORDS and POS:

# W[0]             = lemma (where w[1] is the similarity)
# if W[0] in model = if the wordVector is present in the model
# [:x]             = the slice 
valid_top_words = [w[0] for w in most_common if w[0] in model.wv][:50000]

with open("Z_matrix_words.json", "w", encoding="utf-8") as f:
    json.dump(valid_top_words, f) #LIST 

# for every word in valid_top_words, 
# if it is in word_to_pos 
# add it to the DICTIONARY word_to_pos_filtered
# es. {'cane': 'N'}
word_to_pos_filtered = {w: word_to_pos[w] for w in valid_top_words if w in word_to_pos}

with open("word_to_pos.json", "w", encoding="utf-8") as f:
    json.dump(word_to_pos_filtered, f) 


# 3. EXTRACT WORD VECTORS (L2 normalization)

# a) gets the word vetors of our valid_top_words
# b) calculates the length of the vectors
# c) L2 normalization, divide each vector by its length
# d) 1e-9 epsilon to avoid division by zero

print(f"Extracting vectors for {len(valid_top_words)} words...")
word_vectors = model.wv[valid_top_words].astype('float32') 
length = np.linalg.norm(word_vectors, axis=1, keepdims=True) 
vecs_norm = word_vectors / (length + 1e-9) 

# 4. GENERATE THE MATRIX 

# a) dot product of all the normalized vectors of the matrix
# b) .T traspose, put the rows in columns to calculate the matrix

print("Calculating Similarity Matrix ...")
similarity_matrix = np.dot(vecs_norm, vecs_norm.T) 

# 5. RESULTS and SAVE
print(f"Matrix Complete!")
print(f"Shape: {similarity_matrix.shape}")
print(f"Memory Usage: {similarity_matrix.nbytes / (1024**3):.2f} GB")

print("Saving matrix to disk...")
np.save("Z_similarity_matrix_50k.npy", similarity_matrix)

print("Saving vecs_norm for vector arithmetic...")
np.save("Z_vecs_norm_50k.npy", vecs_norm)

print("Done! Files saved: Z_similarity_matrix_50k.npy, Z_vecs_norm_50k.npy and Z_matrix_words.json")
