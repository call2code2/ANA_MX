import os
from gensim.models.callbacks import CallbackAny2Vec
from collections import defaultdict, Counter
from spellchecker import SpellChecker

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

from collections import defaultdict, Counter




