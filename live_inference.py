import os
import tempfile
import random
import logging
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ==========================================
# 0. LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==========================================
# 1. CONFIGURATION & MODEL LOADING
# ==========================================
MODEL_PATH = "stutter_detector_model.h5"
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40
MAX_LEN = 130
THRESHOLD = 0.70  # Increased threshold to reduce false positives

# Initialize FastAPI
app = FastAPI(title="Stutter Detection AI")

# Load Model Globally
try:
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"⚠️ Error loading model: {e}")
    model = None

# ==========================================
# 2. FEATURE EXTRACTION LOGIC
# ==========================================
def extract_features(file_path):
    """
    Extracts 120 features: 40 Base MFCCs + 40 Deltas + 40 Delta-Deltas.
    Exactly matches the training script preprocessing.
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # 1. Base MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        
        # 2. Delta MFCCs
        mfccs_delta = librosa.feature.delta(mfccs)
        
        # 3. Delta-Delta MFCCs
        mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

        # Combine all 120 features
        features = np.vstack([mfccs, mfccs_delta, mfccs_delta2])

        # Pad or truncate to MAX_LEN
        if features.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - features.shape[1]
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :MAX_LEN]

        # Transpose to shape (Time, Features) -> (130, 120)
        return features.T.astype(np.float32)
    except Exception as e:
        logger.error(f"Extraction error: {e}", exc_info=True)
        return None

def get_therapy_feedback(lang='en'):
    strategies = {
        'en': ["Take a deep breath.", "Try an easy onset.", "Slow down your speech."],
    }
    return random.choice(strategies.get(lang, strategies['en']))

# ==========================================
# 3. FASTAPI ENDPOINTS
# ==========================================
@app.post("/predict")
async def predict_audio(audio: UploadFile = File(...)):
    logger.info(f"Received prediction request. File size: {audio.size} bytes.")
    
    if model is None:
        logger.error("Prediction attempted, but model is not loaded.")
        return JSONResponse(status_code=500, content={"error": "Model not loaded on server."})

    # Save uploaded audio blob to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        content = await audio.read()
        temp_audio.write(content)
        temp_path = temp_audio.name

    try:
        # Extract features
        logger.info(f"Extracting features from temporary audio file: {temp_path}")
        features = extract_features(temp_path)
        if features is None:
            logger.warning("Feature extraction returned None.")
            return JSONResponse(status_code=400, content={"error": "Failed to process audio."})

        # Run Inference
        features = np.expand_dims(features, axis=0) # Shape: (1, 130, 120)
        probability = float(model.predict(features, verbose=0)[0][0])
        
        # Applying the updated threshold
        is_stuttering = probability >= THRESHOLD
        feedback = get_therapy_feedback() if is_stuttering else "Great job! Keep going."

        logger.info(f"Inference complete | Probability: {probability:.4f} | Stuttering: {is_stuttering}")

        return JSONResponse(content={
            "probability": probability,
            "is_stuttering": is_stuttering,
            "feedback": feedback
        })
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")

# ==========================================
# 4. EMBEDDED FRONTEND (HTML/JS/CSS)
# ==========================================
@app.get("/")
def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stutter Detection AI</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            .pulse {
                animation: pulse-animation 2s infinite;
            }
            @keyframes pulse-animation {
                0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
                70% { box-shadow: 0 0 0 20px rgba(59, 130, 246, 0); }
                100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
            }
        </style>
    </head>
    <body class="bg-gray-50 min-h-screen font-sans text-gray-800 flex flex-col items-center justify-center p-4">
        
        <div class="max-w-md w-full bg-white rounded-2xl shadow-xl overflow-hidden p-8 text-center">
            <div class="flex justify-center mb-6">
                <div class="bg-blue-100 text-blue-600 p-4 rounded-full">
                    <i data-lucide="mic" class="w-8 h-8"></i>
                </div>
            </div>
            
            <h1 class="text-2xl font-bold text-gray-900 mb-2">Live Speech Therapy AI</h1>
            <p class="text-sm text-gray-500 mb-8">Press the button below and speak naturally. The AI analyzes 3-second chunks of audio.</p>
            
            <button id="recordBtn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-8 rounded-full shadow-lg transition-all transform hover:scale-105 flex items-center justify-center mx-auto w-48 mb-6">
                <i data-lucide="play" id="btnIcon" class="mr-2"></i>
                <span id="btnText">Start Session</span>
            </button>

            <div id="statusContainer" class="hidden flex-col items-center">
                <div class="text-sm font-semibold text-gray-600 uppercase tracking-widest mb-1">Status</div>
                <div id="statusText" class="text-lg font-medium text-blue-600 mb-4 animate-pulse">Listening...</div>
                
                <div class="w-full bg-gray-100 rounded-lg p-4 text-left border-l-4 border-transparent" id="resultCard">
                    <div class="flex justify-between items-center mb-2">
                        <span class="font-semibold text-gray-700" id="resultLabel">Waiting...</span>
                        <span class="text-xs text-gray-500 font-mono" id="probLabel">--%</span>
                    </div>
                    <p class="text-sm text-gray-600 italic" id="feedbackText">Speak to receive feedback.</p>
                </div>
            </div>
        </div>

        <script>
            // Initialize Icons
            lucide.createIcons();

            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            
            const recordBtn = document.getElementById('recordBtn');
            const btnText = document.getElementById('btnText');
            const btnIcon = document.getElementById('btnIcon');
            const statusContainer = document.getElementById('statusContainer');
            const resultCard = document.getElementById('resultCard');
            const resultLabel = document.getElementById('resultLabel');
            const probLabel = document.getElementById('probLabel');
            const feedbackText = document.getElementById('feedbackText');

            recordBtn.addEventListener('click', toggleRecording);

            async function toggleRecording() {
                if (!isRecording) {
                    await startRecordingSession();
                } else {
                    stopRecordingSession();
                }
            }

            async function startRecordingSession() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    
                    // Update UI
                    isRecording = true;
                    recordBtn.classList.replace('bg-blue-600', 'bg-red-500');
                    recordBtn.classList.replace('hover:bg-blue-700', 'hover:bg-red-600');
                    recordBtn.classList.add('pulse');
                    btnText.innerText = "Stop Session";
                    lucide.createIcons({
                        icons: { square: lucide.icons.square },
                        nameAttr: 'data-lucide',
                        attrs: { class: "mr-2 w-5 h-5", id: "btnIcon" }
                    });
                    statusContainer.classList.remove('hidden');
                    statusContainer.classList.add('flex');

                    recordChunk(stream);

                } catch (err) {
                    alert("Microphone access denied or not available: " + err);
                }
            }

            function recordChunk(stream) {
                if (!isRecording) return;

                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    await sendToAI(audioBlob);
                    
                    // Loop immediately for the next 3 seconds if still recording
                    if (isRecording) {
                        recordChunk(stream);
                    } else {
                        // Cleanup stream tracks when fully stopped
                        stream.getTracks().forEach(track => track.stop());
                    }
                };

                mediaRecorder.start();
                // Record for exactly 3 seconds to match model constraints
                setTimeout(() => {
                    if(mediaRecorder.state === "recording") {
                        mediaRecorder.stop();
                    }
                }, 3000); 
            }

            function stopRecordingSession() {
                isRecording = false;
                
                // Update UI
                recordBtn.classList.replace('bg-red-500', 'bg-blue-600');
                recordBtn.classList.replace('hover:bg-red-600', 'hover:bg-blue-700');
                recordBtn.classList.remove('pulse');
                btnText.innerText = "Start Session";
                document.getElementById('statusText').innerText = "Session Ended";
                document.getElementById('statusText').classList.remove('animate-pulse');
                
                // Switch icon back
                const iconElement = document.getElementById('btnIcon');
                iconElement.innerHTML = ''; // Clear SVG contents
                const newIcon = document.createElement('i');
                newIcon.setAttribute('data-lucide', 'play');
                newIcon.setAttribute('id', 'btnIcon');
                newIcon.className = "mr-2 w-5 h-5";
                iconElement.parentNode.replaceChild(newIcon, iconElement);
                lucide.createIcons();
            }

            async function sendToAI(audioBlob) {
                const formData = new FormData();
                formData.append("audio", audioBlob, "recording.webm");

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        console.error(data.error);
                        return;
                    }

                    updateResultUI(data);

                } catch (err) {
                    console.error("Error communicating with AI backend:", err);
                }
            }

            function updateResultUI(data) {
                probLabel.innerText = (data.probability * 100).toFixed(1) + "%";
                feedbackText.innerText = data.feedback;

                if (data.is_stuttering) {
                    resultLabel.innerText = "⚠️ Stuttering Detected";
                    resultCard.classList.remove('bg-gray-100', 'border-transparent', 'bg-green-50', 'border-green-500');
                    resultCard.classList.add('bg-orange-50', 'border-orange-500');
                    resultLabel.classList.replace('text-gray-700', 'text-orange-700');
                } else {
                    resultLabel.innerText = "✅ Fluent";
                    resultCard.classList.remove('bg-gray-100', 'border-transparent', 'bg-orange-50', 'border-orange-500');
                    resultCard.classList.add('bg-green-50', 'border-green-500');
                    resultLabel.classList.replace('text-orange-700', 'text-green-700');
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    logger.info("Starting server... Open http://127.0.0.1:8000 in your browser.")
    uvicorn.run(app, host="127.0.0.1", port=8000)