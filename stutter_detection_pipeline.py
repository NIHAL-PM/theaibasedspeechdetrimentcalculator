# -*- coding: utf-8 -*-
"""
End-to-End Stuttering Detection Pipeline (SEP-28k)
==================================================
A single-file solution that:
1. Auto-downloads dataset labels (SEP-28k).
2. Downloads raw podcast audio.
3. Clips and sorts audio into 'fluent' vs 'stuttering'.
4. Trains a CNN-LSTM model on the real data.
5. Demonstrates the therapy logic.

WARNING: The full dataset is large (20GB+).
By default, DEMO_MODE is True (downloads only 5 episodes).
Set DEMO_MODE = False to train on the full dataset.
"""

import os
import pathlib
import subprocess
import urllib.request
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Input
from sklearn.model_selection import train_test_split
import random

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
DEMO_MODE = False  # ⚠️ Set to False for REAL results (takes longer)
LANGUAGE = 'en'   # 'en', 'es', 'hi', 'fr'
EPOCHS = 20       # Increased for better learning

# Paths
BASE_DIR = "/content"
RAW_WAV_DIR = os.path.join(BASE_DIR, "sep28k_raw_wavs")
DATASET_OUTPUT = os.path.join(BASE_DIR, "dataset")
EPISODES_CSV = os.path.join(BASE_DIR, "SEP-28k_episodes.csv")
LABELS_CSV = os.path.join(BASE_DIR, "SEP-28k_labels.csv")

# URLs for Metadata
URL_EPISODES = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_episodes.csv"
URL_LABELS = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/SEP-28k_labels.csv"

# Audio Params for Model
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40
MAX_LEN = 130

# ==========================================
# 2. UTILITIES & SETUP
# ==========================================
def setup_environment():
    print("\n[1/5] Setting up Environment & Fetching Metadata...")

    # Download CSVs if missing
    if not os.path.exists(EPISODES_CSV):
        print(f"   Downloading {EPISODES_CSV}...")
        try:
            urllib.request.urlretrieve(URL_EPISODES, EPISODES_CSV)
        except Exception as e:
            print(f"   ❌ Error downloading episodes CSV: {e}")

    if not os.path.exists(LABELS_CSV):
        print(f"   Downloading {LABELS_CSV}...")
        try:
            urllib.request.urlretrieve(URL_LABELS, LABELS_CSV)
        except Exception as e:
            print(f"   ❌ Error downloading labels CSV: {e}")

    # Install dependencies if missing
    try:
        import soundfile
    except ImportError:
        print("   Installing soundfile...")
        subprocess.check_call(["pip", "install", "soundfile"])

    print("   ✅ Setup Complete.")

# ==========================================
# 3. DOWNLOADER (Raw Podcasts)
# ==========================================
def download_raw_audio():
    print("\n[2/5] Downloading Audio Episodes...")

    if not os.path.exists(EPISODES_CSV):
        print("❌ Critical Error: CSV not found.")
        return

    # --- FIX: Handle No-Header CSV ---
    try:
        # SEP-28k_episodes.csv has NO header.
        # Columns are: ShowName, EpTitle, Url, Show, EpId
        df = pd.read_csv(EPISODES_CSV, sep=r',\s+', engine='python', header=None)

        # Manually assign correct column names
        df.columns = ['ShowName', 'EpTitle', 'Url', 'Show', 'EpId']

        # Clean up data (remove quotes if any, strip spaces)
        df['Show'] = df['Show'].astype(str).str.strip()
        df['EpId'] = df['EpId'].astype(str).str.strip()

    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return
    # -------------------------------

    if DEMO_MODE:
        print("   ⚠️ DEMO MODE ACTIVE: Downloading only first 5 episodes.")
        df = df.head(5)

    os.makedirs(RAW_WAV_DIR, exist_ok=True)

    for index, row in df.iterrows():
        try:
            show_abrev = row['Show']
            ep_idx = row['EpId']
            episode_url = row['Url']
        except KeyError as e:
            print(f"   ❌ Missing column in row {index}: {e}")
            continue

        episode_dir = pathlib.Path(f"{RAW_WAV_DIR}/{show_abrev}/")
        os.makedirs(episode_dir, exist_ok=True)
        wav_path = pathlib.Path(f"{episode_dir}/{ep_idx}.wav")

        if os.path.exists(wav_path):
            print(f"   Skipping {show_abrev} {ep_idx} (Exists)")
            continue

        print(f"   Processing {show_abrev} Episode {ep_idx}...")

        # Download to temp file
        temp_audio = pathlib.Path(f"{episode_dir}/temp_audio")
        # -q for quiet, -O for output file
        cmd_download = f'wget -O "{temp_audio}" "{episode_url}" -q'

        try:
            subprocess.call(cmd_download, shell=True)

            # Check if file exists and has size
            if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) < 1000:
                print(f"      ⚠️ Download failed or file too small: {episode_url}")
                if os.path.exists(temp_audio): os.remove(temp_audio)
                continue

            # Convert to 16khz mono wav using ffmpeg
            cmd_convert = f'ffmpeg -y -i "{temp_audio}" -ac 1 -ar 16000 "{wav_path}" -hide_banner -loglevel error'
            subprocess.call(cmd_convert, shell=True)

            if os.path.exists(temp_audio):
                os.remove(temp_audio)

        except Exception as e:
            print(f"      ❌ Failed: {e}")

    print("   ✅ Audio Download Phase Complete.")

# ==========================================
# 4. CLIPPER (Cut & Sort)
# ==========================================
def clip_and_sort_audio():
    print("\n[3/5] Clipping and Sorting into Fluent/Stuttering...")

    if not os.path.exists(LABELS_CSV):
        print("❌ Labels CSV not found.")
        return

    # Read Labels CSV
    try:
        df = pd.read_csv(LABELS_CSV, dtype={"EpId": str})
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"❌ Error reading labels CSV: {e}")
        return

    df = df.sort_values(by=['Show', 'EpId', 'Start'])

    os.makedirs(f"{DATASET_OUTPUT}/fluent", exist_ok=True)
    os.makedirs(f"{DATASET_OUTPUT}/stuttering", exist_ok=True)

    current_wav_path = ""
    audio_data = None
    sample_rate = 0

    processed_count = 0

    # Iterate through labels
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Clips"):
        show = row['Show']
        ep_id = str(row['EpId']).strip()
        clip_id = str(row['ClipId'])
        start = int(row['Start'])
        stop = int(row['Stop'])

        wav_path = f"{RAW_WAV_DIR}/{show}/{ep_id}.wav"

        # Optimization: Load file only if changed
        if wav_path != current_wav_path:
            if not os.path.exists(wav_path):
                continue

            try:
                sample_rate, audio_data = wavfile.read(wav_path)
                current_wav_path = wav_path
            except Exception as e:
                continue

        # Classify based on labels
        try:
            is_fluent = row['NoStutteredWords'] >= 2
        except KeyError:
            is_fluent = True

        folder = "fluent" if is_fluent else "stuttering"
        filename = f"{show}_{ep_id}_{clip_id}.wav"
        output_path = f"{DATASET_OUTPUT}/{folder}/{filename}"

        if not os.path.exists(output_path):
            try:
                if audio_data is not None:
                    clip_data = audio_data[start:stop]
                    wavfile.write(output_path, sample_rate, clip_data)
                    processed_count += 1
            except Exception:
                pass

    print(f"   ✅ Clipping Complete. {processed_count} clips generated.")

# ==========================================
# 5. MODEL TRAINING
# ==========================================
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

        if mfccs.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :MAX_LEN]
        return mfccs.T
    except Exception as e:
        return None

def run_training():
    print("\n[4/5] Training Model...")

    X = []
    y = []

    # Load Data
    categories = {'fluent': 0, 'stuttering': 1}
    valid_data_found = False

    for category, label in categories.items():
        folder_path = os.path.join(DATASET_OUTPUT, category)
        if not os.path.exists(folder_path): continue

        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        print(f"   Loading {len(files)} {category} samples...")

        for file in files:
            feat = extract_features(os.path.join(folder_path, file))
            if feat is not None:
                X.append(feat)
                y.append(label)
                valid_data_found = True

    if not valid_data_found:
        print("❌ No training data found. Did the download/clipping work?")
        return None

    X = np.array(X)
    y = np.array(y)

    if len(X) < 5:
        print("⚠️ Not enough data to train (need at least 5 samples).")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build Model
    model = Sequential()
    # --- FIX: Using explicit Input layer to avoid warning ---
    model.add(Input(shape=(MAX_LEN, N_MFCC)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # -------------------------------------------------------
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train
    print(f"   Training for {EPOCHS} epochs...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_test, y_test))
    print("   ✅ Model Trained.")
    return model

# ==========================================
# 6. THERAPY LOGIC & EXECUTION
# ==========================================
def therapy_feedback(prediction_prob, lang='en'):
    strategies = {
        'en': ["Take a deep breath.", "Easy Onset: start gently.", "Slow down."],
        'es': ["Respira profundo.", "Inicio Suave.", "Reduce la velocidad."],
        'hi': ["Gehri saans lein.", "Dheere bole.", "Shuruat aaram se karein."],
        'fr': ["Respirez profondément.", "Commencez doucement.", "Ralentissez."]
    }
    msgs = strategies.get(lang, strategies['en'])

    if prediction_prob > 0.5:
        return f"⚠️ Detected ({prediction_prob:.2f}) -> 🤖 BOT: {random.choice(msgs)}"
    return f"✅ Fluent ({prediction_prob:.2f})"

def main():
    setup_environment()
    download_raw_audio()
    clip_and_sort_audio()
    model = run_training()

    if model:
        print("\n[5/5] Running Inference Demo on Test Sample...")
        fake_input = np.random.normal(0, 1, SAMPLE_RATE*DURATION)
        sf.write('temp_demo.wav', fake_input, SAMPLE_RATE)

        feat = extract_features('temp_demo.wav')
        if feat is not None:
            feat = np.expand_dims(feat, axis=0)
            pred = model.predict(feat)[0][0]
            print(f"\nRESULT: {therapy_feedback(pred, lang=LANGUAGE)}")

        if os.path.exists('temp_demo.wav'):
            os.remove('temp_demo.wav')

if __name__ == "__main__":
    main()