import os
import numpy as np
import tensorflow as tf
import pickle
import librosa
from skimage.transform import resize

print("🧠 Loading the Embedder Model...")
embedder = tf.keras.models.load_model('music_embedder.h5', compile=False)

# --- 1. THE AUDIO PROCESSOR ---
def preprocess_audio(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    for i in range(num_chunks):
        start = int(i * (chunk_samples - overlap_samples))
        end = int(start + chunk_samples)
                    
        chunk = audio_data[start:end]
                    
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    # Returns an array of shape (Number_of_Chunks, 150, 150, 1)
    return np.array(data) 

# --- 2. SETUP THE FOLDER ---
# CHANGE THIS PATH to your actual folder with the 20 songs!
music_folder = "./Musicdata" 
database = {}

print(f"\n🎧 Scanning music library in: {music_folder}")
print("Calculating 1200-D coordinates for each song...\n")

# --- 3. RUN THE EXTRACTION ---
for filename in os.listdir(music_folder):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        file_path = os.path.join(music_folder, filename)
        
        try:
            # 1. Get the batch of 4-second chunks
            processed_chunks = preprocess_audio(file_path)
            
            # 2. Get embeddings for EVERY chunk at once (Shape: Num_Chunks, 1200)
            chunk_embeddings = embedder.predict(processed_chunks, verbose=0)
            
            # 3. Average them together into one master "vibe" coordinate (Shape: 1200,)
            final_song_embedding = np.mean(chunk_embeddings, axis=0)
            
            # 4. Save it to our database
            database[filename] = final_song_embedding
            print(f"✅ Mapped: {filename} (Averaged {len(processed_chunks)} chunks)")
            
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")

# --- 4. SAVE THE DATABASE ---
with open('music_database.pkl', 'wb') as f:
    pickle.dump(database, f)

print("\n💾 Database built and saved as 'music_database.pkl'!")