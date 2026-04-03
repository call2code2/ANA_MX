import json
import numpy as np

print("Loading Sandbox...")

# 1. Load the words and build the "Address Book"
with open("Z_matrix_words.json", "r", encoding="utf-8") as f:
    words = json.load(f)
word_to_id = {word: i for i, word in enumerate(words)}

# 2. Load the coordinates
vecs_norm = np.load("Z_vecs_norm_50k.npy") # Use your actual filename here
print(f"Loaded {len(words)} words into the sandbox.\n")
print("="*50)
print("Type 4 words separated by spaces: A B C D")
print("Example: italia roma francia parigi")
print("Type 'q' to quit.")
print("="*50)

# 3. The Interactive Loop
while True:
    user_input = input("\nAnalogy > ").strip().lower()
    
    if user_input == 'q':
        print("Exiting Sandbox...")
        break
        
    parts = user_input.split()
    if len(parts) != 4:
        print("Error: You must enter exactly 4 words.")
        continue
        
    wA, wB, wC, wD = parts
    
    # 4. Check if all words exist in the dictionary
    missing_words = [w for w in (wA, wB, wC, wD) if w not in word_to_id]
    if missing_words:
        print(f"Error: These words are not in the top 50k: {missing_words}")
        continue
        
    # 5. Get the IDs and Vectors
    idA, idB, idC, idD = [word_to_id[w] for w in (wA, wB, wC, wD)]
    vA, vB, vC, vD = vecs_norm[idA], vecs_norm[idB], vecs_norm[idC], vecs_norm[idD]
    
    # 6. Calculate the Jumps (Relationship Vectors)
    v_link_1 = vA - vB
    v_link_2 = vC - vD
    
    # 7. Normalize the jumps so we can get the pure Cosine Similarity
    norm_1 = np.linalg.norm(v_link_1)
    norm_2 = np.linalg.norm(v_link_2)
    
    if norm_1 == 0 or norm_2 == 0:
        print("Error: Words are identical, cannot compute jump.")
        continue
        
    v_link_1_norm = v_link_1 / norm_1
    v_link_2_norm = v_link_2 / norm_2
    
    # 8. Calculate the Score!
    score = np.dot(v_link_1_norm, v_link_2_norm)

    # 8. Calculate the Score (Cosine Similarity)
    score = np.dot(v_link_1_norm, v_link_2_norm)
    
    # 9. Calculate the Angle!
    # np.clip prevents math errors if the score accidentally hits 1.0000001 due to floating point rounding
    safe_score = np.clip(score, -1.0, 1.0) 
    
    # Arccosine gives the angle in Radians, so we convert it to Degrees
    angle_in_radians = np.arccos(safe_score)
    angle_in_degrees = np.degrees(angle_in_radians)
    
    print(f"[{wA} : {wB}] :: [{wC} : {wD}]")
    print(f"Relational Score: {score:.4f}")
    print(f"Angle Between Jumps: {angle_in_degrees:.1f} degrees")
    
    