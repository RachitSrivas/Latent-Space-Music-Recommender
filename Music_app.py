import os
# Flip the switch to use the legacy Keras version!
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import gdown

# --- DOWNLOAD CODE ---
MODEL_PATH = "Trained_model.h5"
# This is the exact ID from your Google Drive link
DRIVE_FILE_ID = "1hoL_DLAnycV9uoQDaFMXUFGHlv6y85ys" 

@st.cache_resource 
def fetch_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model from Google Drive... Please wait a moment."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return True

# Run the download check before you do anything else
fetch_model()

# --- MODEL LOADING ---
@st.cache_resource()
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# --- PREPROCESSING ---
def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    chunk_duration = 4  # seconds
    overlap_duration = 2  # seconds
                
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
        chunk = audio_data[start:end]
                    
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)
    
    return np.array(data)

# --- PREDICTION ---
def model_prediction(X_test):
    # FIXED: Actually load the model instead of just fetching the file!
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]

# --- UI DASHBOARD ---
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction"])

## Main Page
if app_mode == "Home":
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #181646;  /* Blue background */
        color: white;
    }
    h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.markdown(''' ## Welcome to the,\n
    ## Music Genre Classification System! 🎶🎧''')
    
    st.markdown("""
**Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.

### Why Choose Us?
- **Accuracy:** Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- **User-Friendly:** Simple and intuitive interface for a smooth user experience.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Prediction** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!
""")

# About Project
elif app_mode == "About Project":
    st.markdown("""
                ### About Project
                Music. Experts have been trying for a long time to understand sound and what differentiates one song from another. How to visualize sound. What makes a tone different from another.

                This data hopefully can give the opportunity to do just that.

                ### About Dataset
                #### Content
                1. **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds.
                2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.
                3. **images original** - A visual representation for each audio file. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. **CSV files** - Containing features of the audio files for further analysis.
                """)

# Prediction Page
elif app_mode == "Prediction":
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    
    if test_mp3 is not None:
        # FIXED: Save the uploaded file temporarily so librosa can safely read it
        filepath = "temp_audio_file." + test_mp3.name.split('.')[-1]
        with open(filepath, "wb") as f:
            f.write(test_mp3.getbuffer())

        # Show Button
        if st.button("Play Audio"):
            st.audio(test_mp3)
        
        # Predict Button
        if st.button("Predict"):
            with st.spinner("Analyzing audio... Please Wait.."):       
                X_test = load_and_preprocess_data(filepath)
                result_index = model_prediction(X_test)
                st.balloons()
                label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
                st.markdown("**:blue[Model Prediction:] It's a  :red[{}] music**".format(label[result_index]))