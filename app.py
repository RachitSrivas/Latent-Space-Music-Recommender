import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import librosa
from skimage.transform import resize
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- CONFIGURATION ---
MUSIC_FOLDER = "./Musicdata" 
# Safety check: Ensure the folder exists so the app never crashes on save
os.makedirs(MUSIC_FOLDER, exist_ok=True)

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Music Recommender", page_icon="🎧", layout="centered")

st.title("🎧 Latent Space Music Recommender")
st.markdown("""
Upload a song, and this custom AI will map its acoustic DNA into a 1200-dimensional Latent Space. 
If the AI hasn't heard the song before, it will dynamically learn it and expand its universe!
""")
st.divider()

# --- 2. LOAD THE BRAIN AND DATABASE ---
# We cache the AI model because it is heavy and never changes
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model('music_embedder.h5', compile=False)

# We do NOT cache the database because we want it to dynamically update!
def load_database():
    if os.path.exists('music_database.pkl'):
        with open('music_database.pkl', 'rb') as f:
            return pickle.load(f)
    return {} # Return empty if it doesn't exist yet

try:
    embedder = load_ai_model()
    database = load_database()
except Exception as e:
    st.error(f"Error loading AI model. Details: {e}")
    st.stop()

# --- 3. THE AUDIO PROCESSOR ---
def get_embedding(file_path):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    # Calculate how many chunks we can slice
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    # Slice the audio and convert to Spectrograms
    for i in range(num_chunks):
        start = int(i * (chunk_samples - overlap_samples))
        end = int(start + chunk_samples)
        chunk = audio_data[start:end]
        
        mel = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel = resize(np.expand_dims(mel, axis=-1), (150, 150))
        data.append(mel)
        
    processed_chunks = np.array(data)
    
    # Push chunks through the AI
    chunk_embeddings = embedder.predict(processed_chunks, verbose=0)
    
    # Average the chunks into one master 1200-D coordinate
    return np.mean(chunk_embeddings, axis=0)

# --- 4. THE UI & LOGIC ---
uploaded_file = st.file_uploader("Upload an MP3 or WAV track to analyze...", type=["mp3", "wav"])

if uploaded_file is not None:
    st.markdown("### 🎵 Uploaded Track:")
    st.audio(uploaded_file)
    
    if st.button("🔮 Map DNA & Find Similar Songs", use_container_width=True):
        with st.spinner("Analyzing the Latent Space..."):
            
            # Define where this file will permanently live
            permanent_path = os.path.join(MUSIC_FOLDER, uploaded_file.name)
            
            try:
                # --- THE UPGRADE: DYNAMIC EXPANSION ---
                if uploaded_file.name not in database:
                    st.toast("✨ New song detected! Adding to the Latent Space...")
                    
                    # 1. Save the physical audio file permanently
                    with open(permanent_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                        
                    # 2. Calculate the 1200-D coordinate
                    target_embedding = get_embedding(permanent_path)
                    
                    # 3. Update our database in memory
                    database[uploaded_file.name] = target_embedding
                    
                    # 4. Overwrite the .pkl file on disk so it remembers for next time
                    with open('music_database.pkl', 'wb') as f:
                        pickle.dump(database, f)
                else:
                    st.toast("Recognized song! Pulling from existing database...")
                    # If we already know the song, just pull its math from the dictionary
                    target_embedding = database[uploaded_file.name]

                # --- SEARCH THE DATABASE ---
                target_reshaped = target_embedding.reshape(1, -1)
                recommendations = []
                
                for song_name, embedding in database.items():
                    # Skip the uploaded song so it doesn't recommend itself as #1
                    if song_name == uploaded_file.name:
                        continue
                        
                    compare_reshaped = embedding.reshape(1, -1)
                    score = cosine_similarity(target_reshaped, compare_reshaped)[0][0]
                    recommendations.append((song_name, score))
                    
                # Sort by highest match score
                recommendations.sort(key=lambda x: x[1], reverse=True)
                
                st.divider()
                st.success("Analysis Complete!")
                st.subheader("🎧 Your Top Recommended Queue:")
                
                # --- PRINT THE TOP 10 AND PLAY THEM ---
                for i in range(min(10, len(recommendations))):
                    song, score = recommendations[i]
                    percentage = score * 100
                    
                    with st.container(border=True):
                        if percentage > 85:
                            st.markdown(f"#### {i+1}. {song} — :green[`{percentage:.2f}% Match`]")
                        else:
                            st.markdown(f"#### {i+1}. {song} — `{percentage:.2f}% Match`")
                        
                        song_path = os.path.join(MUSIC_FOLDER, song)
                        if os.path.exists(song_path):
                            st.audio(song_path)
                        else:
                            st.warning(f"Audio file not found at {song_path}")
                        
            except Exception as e:
                st.error(f"Failed to process the audio file: {e}")