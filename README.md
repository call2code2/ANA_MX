# Analogy Matrix Project

This project focuses on generating an Analogy Matrix using Word2Vec models trained on the PAISA corpus. The project pipeline is divided into 5 different scripts, each serving a necessary purpose.

##  Project Structure

In order of use, the scripts in this project are:

* **`data_utils`**: Contains useful functions used in the other scripts.
* **`train`**: Trains the Word2Vec model.
* **`gen_simX`**: Creates a similarity matrix.
* **`gold_extraction`**: Chooses the section of interest in the matrix.
* **`ana_mat`**: Handles the actual creation of the analogy matrix.

---

##  Workflow & Usage

### 1. Model Training (`train.py`)
Trains a Word2Vec model on the PAISA corpus.

* **Data Loading**: Loads database sentences using `SentenceIterator`.
* **Parameters**: Uses 300 dimensions, skipgram, 10 window, 15 negative (sampling), 75 min_count, and 15 epochs.
* **Hardware**: Detects the number of cores that can be used during training, leaving out 2 cores.
* **Modes**: Requires the user to choose the training mode between raw words or lemmas. Also prompts the user to overwrite if the input model name already exists.
* **Output**: Saves the trained model.

### 2. Similarity Matrix Generation (`gen_simX.py`)
Generates a similarity matrix using the trained Word2Vec model.

* **Data Preparation**: Loads and stores the most common words and word-to-POS mappings into variables (`Z_top_words`, `Z_word_to_pos`, `Z_matrix_words`).
* **Extraction & Normalization**: Extracts word vectors (WV) from valid top words. Calculates the length of the vector and applies L2 normalization (dividing each vector by its length so that each $V_l=1$).
* **Matrix Generation**: Creates a 50k x 50k grid. Calculates the similarity matrix using `np.dot`, as the dot product between normalized vectors is equal to cosine similarity.
* **Outputs**: Saves three distinct files:
    * `Z_similarity_matrix_50k.npy`: Contains the value of the matrix.
    * `Z_vecs_norm_50k.npy`: Contains the normalized vectors.
    * `Z_matrix_words.json`: Contains the words of the matrix.

### 3. Section Extraction (`gold_extraction.py`)
Chooses a threshold span from the matrix to isolate relevant pairs.

* **Threshold Filtering**: Loops through the matrix in chunks to filter the pairs accepted for the final CSV. Drops unrelated word pairs (similarity < 0.35) and perfect synonyms (similarity > 0.9).
* **Deduplication**: Applies a mask that keeps just the upper triangle of the matrix to avoid duplicates (A-B, B-A).
* **Outputs**: Saves the final results in a Pandas CSV.

### 4. Analogy Matrix (`ana_mat.py`)
Executes the primary analogy search and matrix creation.

* **Load Data**: Uses precalculated data from the previous scripts. This includes `Z_matrix_words.json` (ordered list of matrix words), `word_to_pos.json` (dictionary of words and associated most common POS), `Z_vecs_norm_50k.json` (normalized vectors), and `gold_pairs_50k.csv` (pairs extracted from the original matrix).
* **Generate V-Links**: Pre-allocates memory for V-links, gets vector indices from `df_golden` for every normalized vector, and calculates the vlink: $V_a - V_b$.
* **Analogy Search**: Employs "Double tiled batching" to fetch 1k rows and 50k columns at a time, calculating the 1k x 50k matrix using `np.dot`.
* **Candidate Ranking**: A "Bouncer" mechanism finds the k-best tile candidates and returns the local indices using `.argpartition`.
* **Translation & Processing**: A translation loop translates local indices to global indices. Loads rows of `df_golden` into pairs 1,2 (A:B) and (C:D). Gets the original matrix index to find associated matrix words and extracts vector norms for the already sorted indices.
* **Final Output**: Applies filters (stopwords, close synonym, count), pushes to the heap, and saves the final output.

### 5. data_utils.py

#### a. Memory-Efficient Iterator (`SentenceIterator`)
To handle large datasets without exceeding RAM limits, the `SentenceIterator` class processes the database sequentially. It keeps only a single sentence in memory at any given time.
* **Smart Parsing:** Automatically skips commented lines (starting with `#`) and recognizes empty lines as sentence boundaries.
* **Punctuation Filtering:** Automatically drops tokens tagged as `PUNCT`.
* **Flexible Extraction:** Includes a toggle (`use_lemma`) to yield either the base lemma or the raw word from the text corpus. All yielded tokens are automatically lowercased.

#### b. Vocabulary Validation (`is_real_word`)
Before a word is counted, it passes through a strict validation function to ensure data quality. To be considered a "real word", a token must:
* Be longer than a single letter.
* Contain only alphabetic characters (dropping symbols, numbers, etc.).
* Be recognized as a valid Italian word by the `pyspellchecker` library.

#### c. Frequency & POS Aggregation (`get_common_words_with_pos`)
This function extracts the higer frequency words from the dataset, while also resolving POS ambiguity.
* **Targeted Filtering:** Only processes words that match specific target POS tags (e.g., 'S', 'V', 'A') and pass the `is_real_word` check.
* **Counting:** Utilizes Python's `defaultdict` to safely aggregate counts without `KeyError` exceptions, and `Counter` to easily extract the top *N* most common words.
* **Dominant POS Resolution:** If a lemma appears with multiple POS tags across the corpus, the script tracks the frequencies of each tag and assigns the word the tag it appears with most frequently. 
* **Output:** Returns the top *N* words and a dictionary mapping each word to its dominant POS tag.




##  Libraries
* NumPy
* Pandas
* Gensim (for Word2Vec)
* collection
* spellchecker
* multiprocessing
* json
* heapq
