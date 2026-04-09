# 🎧 Latent Space Music Recommender

> An AI-powered content-based music recommendation system that understands the **actual sound and vibe of a song** using Deep Learning, Latent Space Embeddings, and Cosine Similarity.

## 📌 Description

This project is an extended and more advanced version of our previous **Music Genre Classification System**.  

In the earlier project, we built a Deep Learning model using CNNs to classify songs into different genres based on Mel-Spectrogram features.  

Building upon that foundation, we extended the same trained CNN architecture beyond classification by removing its final dense layer and using it as a **feature extractor**. This allowed us to generate **latent space embeddings** that capture the deeper acoustic characteristics of songs.  

Using these embeddings along with **Cosine Similarity**, we transformed the classification model into a powerful **content-based music recommendation engine** that suggests songs based on actual sound, vibe, and musical patterns.  

This project demonstrates how an existing ML model can be scaled into a real-world intelligent product with enhanced functionality and practical usability.

---

## 🔄 Project Evolution: From Genre Classification to Music Recommendation

This project was developed by extending our previous **Music Genre Classification System** into a more practical and intelligent recommendation engine.

Instead of building a new model from scratch, we reused and upgraded our trained CNN model through a structured 3-step pipeline:

### 🩺 Step 1: Model Surgery (`step1surgery.py`)
- Loaded the previously trained CNN genre classification model
- Removed the final dense classification layer
- Converted the model into a **headless feature extractor**
- Enabled extraction of **1200-dimensional latent embeddings**

### 🗂️ Step 2: Database Creation (`step2database.py`)
- Processed all existing songs in the dataset
- Split each track into 4-second chunks
- Converted chunks into Mel-Spectrograms
- Generated embeddings for every chunk
- Averaged chunk embeddings into one master vector per song
- Stored all vectors in a local **Pickle database (`.pkl`)**

### 🎯 Step 3: Recommendation Engine (`step3recommender.py`)
- Built the similarity search pipeline using **Cosine Similarity**
- Compared uploaded song embeddings with stored database vectors
- Ranked songs based on acoustic closeness
- Returned the **Top 10 most similar recommendations**

### 🚀 Main Application (`app.py`)
The complete system is integrated into an interactive **Streamlit web application**, where users can:

- Upload songs
- View similarity percentages
- Listen to recommended tracks directly

---

## 📈 Dynamic Self-Learning Database

One of the most powerful features of this project is its **self-expanding recommendation space**.

### How it works:
- Whenever a user uploads a song:
  - The system checks whether that song already exists in the database
- If the song is **new**:
  - Its latent embedding is extracted
  - The audio file is stored locally
  - The vector database is updated permanently

### Why this matters:
- The recommendation engine becomes smarter over time
- The song library keeps expanding automatically
- Future users get:
  - More accurate recommendations
  - Better diversity
  - Improved similarity matches

This makes the project not just a static ML model, but a **continuously improving AI-powered music ecosystem**.



## 🚀 Overview

**Latent Space Music Recommender** is an intelligent acoustic search engine that recommends songs based on their **sonic similarity**, rather than relying on:

- ❌ Human-assigned genre labels  
- ❌ User listening history  
- ❌ Collaborative filtering  

Instead, the system analyzes the **raw audio frequencies** of a track, converts them into mathematical representations, and finds the most similar songs from its database.

This enables highly accurate recommendations based on:

- Energy  
- Rhythm  
- Mood  
- Instrumental texture  
- Overall vibe  

---

## 🧠 Problem Statement

Traditional music recommendation systems often fail because:

- Genre labels are inconsistent and subjective  
- New users face the cold-start problem  
- Recommendations depend heavily on listening history  

This project solves these issues by building a **purely content-based recommendation engine** that understands music at the audio signal level.

---

## 🏗️ System Architecture

### 1. Audio Preprocessing & Feature Extraction

- Input songs are uploaded in `.mp3` or `.wav` format
- Each audio file is divided into **4-second chunks**
- Each chunk is converted into a **Mel-Spectrogram** using **Librosa**

### Why Mel-Spectrogram?
Mel-Spectrograms transform audio into a visual frequency representation that captures:

- Pitch
- Timbre
- Rhythm patterns

This makes it ideal for CNN-based audio understanding.

---

### 2. Deep CNN Feature Learning

A custom **Convolutional Neural Network (CNN)** is used to learn high-level music patterns.

### CNN learns:
- Beat structure
- Vocal texture
- Instrument layers
- Frequency signatures

The CNN was originally trained for **music genre classification**.

---

### 3. Latent Space Embedding Generation

Instead of using the CNN only for classification:

- The final dense classification layer is removed (**headless CNN architecture**)
- The model outputs **1200-dimensional latent feature embeddings**

These embeddings act as the song’s:

- Acoustic fingerprint  
- Sonic DNA  
- Mathematical identity  

---

### 4. Vibe Averaging

Since each song has multiple 4-second chunks:

- Each chunk generates a separate embedding
- All embeddings are averaged into a **single master embedding vector**

This creates a stable representation of the full track’s overall vibe.

---

### 5. Similarity Search Engine

When a user uploads a target song:

- The system extracts its embedding
- Compares it against all stored song embeddings
- Uses **Cosine Similarity** to measure closeness

### Output:
- Top 10 most similar songs
- Match percentages
- Playable recommendations

---

## ✨ Key Features

### 🎯 Content-Based Recommendation
Recommends songs based on actual sound characteristics instead of metadata.

### 🔥 Latent Space Acoustic Search
Finds songs with similar vibe, energy, and feel.

### 📈 Dynamic Self-Expanding Database
If a user uploads a new song:

- Embeddings are automatically extracted
- Song file is stored locally
- Database updates in real time

This creates a continuously growing recommendation ecosystem.

### 🌍 Genre-Agnostic Matching
Can connect songs across genres if they sound similar.

**Example:**
A high-energy Bollywood dance track may match:
- EDM
- Rock
- Pop anthems  

### 🎵 Interactive Streamlit Interface
Users can:

- Upload songs
- View recommendation scores
- Play suggested tracks directly in browser

---
## 🚀 Quick Start (Run Locally)

Want to test the AI on your own computer? Follow these steps:

### **1. Clone the Repository**

 git https://github.com/RachitSrivas/Latent-Space-Music-Recommender
cd Latent-Space-Music-Recommender

### 2. Install the Required Libraries

Make sure Python is installed, then run:
pip install -r requirements.txt

### 3. Set Up the Database

Before running the app, generate the initial vector database using the starter pack of songs:
python step2database.py

### 4. Launch the Web App
streamlit run app.py

The interactive UI will open automatically in your browser.

--

## 🛠️ Tech Stack

### Deep Learning
- TensorFlow
- Keras

### Audio Processing
- Librosa

### Mathematical Similarity Search
- NumPy
- Scikit-learn

### Frontend / UI
- Streamlit

### Database / Storage
- Python Pickle (`.pkl`) for local vector storage

---



## 📸 User Workflow

### Step 1:
Upload a song file

### Step 2:
System extracts:
- Mel-Spectrogram chunks
- CNN embeddings

### Step 3:
Latent vector generated

### Step 4:
Cosine similarity search performed

### Step 5:
Top 10 recommended songs displayed

---

## 📊 Future Improvements

- 🎤 Mood / emotion-based recommendations  
- 🌐 Cloud deployment  
- 🎶 Spotify API integration  
- ❤️ Playlist generation  
- 📱 Mobile-friendly UI  
- ⚡ Faster ANN search using FAISS  

---

## 💡 What Makes This Project Stand Out?

### Strong ML + Product Thinking
This is not just a model — it's a full end-to-end AI product.

### Real Deep Learning Application
Uses:
- CNN feature extraction  
- Transfer learning concepts  
- Latent space representations  

### Scalable System Design
Supports:
- Dynamic database growth  
- Real-time embedding updates  

### Resume / Portfolio Value
Demonstrates:
- Machine Learning expertise  
- Audio signal processing  
- Recommendation systems  
- Full-stack deployment skills  

---

## 👨‍💻 Author

**Rachit Srivastava**

- Passionate about AI, ML, and building impactful products  
- Exploring Deep Learning, Recommendation Systems, and Full-Stack Development  

---

## ⭐ If you like this project

Give it a star on GitHub and feel free to contribute!