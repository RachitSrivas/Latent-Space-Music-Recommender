import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("📂 Loading Music Database...")
with open('music_database.pkl', 'rb') as f:
    database = pickle.load(f)

# --- 1. PICK A TARGET SONG ---
# Let's test it with one of your Jazz tracks! 
# (You can change this string to 'Kala Chashma - NaaSongs.mp3' later to see what it recommends!)
target_song = "Kala Chashma - NaaSongs.mp3"

if target_song not in database:
    print(f"❌ Error: {target_song} not found in database.")
    exit()

# Get the 1200-D coordinate for our target song
target_embedding = database[target_song]

print(f"\n🎧 Target Song: {target_song}")
print("🔍 Searching the Latent Space for similar tracks...\n")

# --- 2. CALCULATE DISTANCES ---
recommendations = []

for song_name, embedding in database.items():
    # Skip the target song itself (otherwise it will be a 100% match with itself)
    if song_name == target_song:
        continue
    
    # Cosine Similarity expects 2D arrays, so we reshape from (1200,) to (1, 1200)
    target_reshaped = target_embedding.reshape(1, -1)
    compare_reshaped = embedding.reshape(1, -1)
    
    # Calculate the similarity score
    score = cosine_similarity(target_reshaped, compare_reshaped)[0][0]
    
    # Add to our list
    recommendations.append((song_name, score))

# --- 3. SORT AND DISPLAY RESULTS ---
# Sort the list by score in descending order (highest score first)
recommendations.sort(key=lambda x: x[1], reverse=True)

print("🎵 TOP 3 RECOMMENDATIONS:")
print("-" * 30)

# Print the top 3 closest matches
for i in range(10):
    song, match_score = recommendations[i]
    # Convert the decimal score to a clean percentage
    percentage = match_score * 100
    print(f"{i+1}. {song}  --> {percentage:.2f}% Match")