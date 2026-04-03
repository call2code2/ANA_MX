import os
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from data_utils import SentenceIterator
import multiprocessing

#  SECTIONS:
# 1) PRETRAIN PARAMETERS (cores, model name, overwrite, training mode)
# 2) TRAIN (Sentence iterator, model parameters)
# 3) SAVE MODEL


DATA_FILE = r"c:\Users\Utente\Desktop\VSCODE\Paisa\paisa.annotated.CoNLL.utf8"

def main():
    print("--- MODEL TRAINER ---")




    #------------------   1) PRETRAIN PARAMETERS   ------------------
    
    # --- a) HARDWARE DETECTION ---
    cores = multiprocessing.cpu_count()
    optimal_workers = max(1, cores - 2) 


    # ---b) MODEL NAME ---
    model_name = input("1. Name your new model: ").strip()
    filename = f"{model_name}.model"
    
    if os.path.exists(filename):
        check = input(f"  '{filename}' exists. Overwrite? (y/n): ").lower()
        if check != 'y': return

    # --- d) CHOOSE TRAINING MODE 
    print("\n2. Select Training Mode:")
    print("   [L] Lemmas (Better for synonyms/concepts: 'gatto' matches 'gatti')")
    print("   [W] Raw Words (Better for grammar: 'gatti' != 'gatto')")
    mode = input("   Choice (L/W): ").strip().lower()
    
    use_lemma = True if mode == 'l' else False






    # -----------------------  2) TRAIN   ---------------------------------------------
    print(f"\n Training '{model_name}' (Mode: {'Lemmas' if use_lemma else 'Raw Words'})...")
    
    # --- a) PASS use_lemma to ITERATOR ---
    sentences = SentenceIterator(DATA_FILE, use_lemma=use_lemma)
    
    # --- b) MODEL PARAMETERS ---

    model = Word2Vec( 
        vector_size=300, 
        window=10, 
        sg=1,
        negative=15,
        min_count=75, 
        sample=6e-5, #gives less importance to articles and prepositions that appears frequently
        workers=optimal_workers,
        epochs=15, 
        
    )



    # --------------------3) SAVE MODEL ----------------------------
    model.save(filename)
    print(f"\n Saved model to: {filename}")

    

if __name__ == "__main__":
    
    main()