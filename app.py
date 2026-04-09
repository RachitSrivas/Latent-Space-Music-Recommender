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
os.makedirs(MUSIC_FOLDER, exist_ok=True)

# Make sure this matches the EXACT alphabetical order of your original training data
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Music Recommender", page_icon="🎧", layout="centered")

st.title("🎧 Latent Space Music Engine")
st.markdown("""
Upload a track to this dual-engine AI. 
**Brain 1** will classify its acoustic genre, and **Brain 2** will map its DNA into a 1200-dimensional Latent Space to find the closest matches in the database!
""")
st.divider()

# --- 2. LOAD THE DUAL BRAINS AND DATABASE ---
@st.cache_resource
def load_ai_models():
    # Load BOTH brains into server memory
    classifier = tf.keras.models.load_model('Trained_model.h5', compile=False)
    embedder = tf.keras.models.load_model('music_embedder.h5', compile=False)
    return classifier, embedder

def load_database():
    if os.path.exists('music_database.pkl'):
        with open('music_database.pkl', 'rb') as f:
            return pickle.load(f)
    return {}

try:
    classifier, embedder = load_ai_models()
    database = load_database()
except Exception as e:
    st.error(f"Error loading AI models. Details: {e}")
    st.stop()

# --- 3. THE UNIFIED AUDIO PROCESSOR ---
def process_and_predict(file_path):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
    
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    
    for i in range(num_chunks):
        start = int(i * (chunk_samples - overlap_samples))
        end = int(start + chunk_samples)
        chunk = audio_data[start:end]
        
        mel = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel = resize(np.expand_dims(mel, axis=-1), (150, 150))
        data.append(mel)
        
    processed_chunks = np.array(data)
    
    # --- BRAIN 1: THE CLASSIFIER ---
    genre_predictions = classifier.predict(processed_chunks, verbose=0)
    average_probabilities = np.mean(genre_predictions, axis=0)
    winning_genre_index = np.argmax(average_probabilities)
    predicted_genre = GENRES[winning_genre_index]
    confidence = average_probabilities[winning_genre_index] * 100
    
    # --- BRAIN 2: THE EMBEDDER ---
    chunk_embeddings = embedder.predict(processed_chunks, verbose=0)
    final_embedding = np.mean(chunk_embeddings, axis=0)
    
    return predicted_genre, confidence, final_embedding

# --- 4. THE UI & LOGIC ---
uploaded_file = st.file_uploader("Upload an MP3 or WAV track to analyze...", type=["mp3", "wav"])

if uploaded_file is not None:
    st.markdown("### 🎵 Uploaded Track:")
    st.audio(uploaded_file)
    
    if st.button("🔮 Analyze Track & Find Matches", use_container_width=True):
        with st.spinner("Processing audio through Dual AI engines..."):
            
            permanent_path = os.path.join(MUSIC_FOLDER, uploaded_file.name)
            
            try:
                # 1. Save the physical audio file permanently (if it's not already there)
                if not os.path.exists(permanent_path):
                    with open(permanent_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                
                # 2. Run the Unified Pipeline
                predicted_genre, confidence, target_embedding = process_and_predict(permanent_path)
                
                # 3. Dynamic Expansion Logic
                if uploaded_file.name not in database:
                    st.toast("✨ New song detected! Adding to the Latent Space...")
                    database[uploaded_file.name] = target_embedding
                    with open('music_database.pkl', 'wb') as f:
                        pickle.dump(database, f)
                else:
                    st.toast("Recognized song! Pulling from existing database...")

                # 4. Search the Database
                target_reshaped = target_embedding.reshape(1, -1)
                recommendations = []
                
                for song_name, embedding in database.items():
                    if song_name == uploaded_file.name:
                        continue
                    compare_reshaped = embedding.reshape(1, -1)
                    score = cosine_similarity(target_reshaped, compare_reshaped)[0][0]
                    recommendations.append((song_name, score))
                    
                recommendations.sort(key=lambda x: x[1], reverse=True)
                
                # --- PRINT RESULTS ---
                st.divider()
                st.success("Analysis Complete!")
                
                # Show Classification prominently
                st.markdown(f"### 🎯 AI Classification: **{predicted_genre.capitalize()}** `({confidence:.1f}% confidence)`")
                
                st.subheader("🎧 Your Top Recommended Queue:")
                
                # Print Top 10
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