
<div align="center">
  <img src="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/mic.svg" width="80" height="80" alt="Stutter Detection AI Logo" />

  # Stuttering Detection AI Web App
  
  ### *Real-Time Speech Therapy & Acoustic Diagnostics via Edge-Optimized Deep Learning*

  [**Explore the Docs**](https://github.com/NIHAL-PM/theaibasedspeechdetrimentcalculator/#readme) • [**Report Bug**](https://github.com/NIHAL-PM/theaibasedspeechdetrimentcalculator/issues) • [**Request Feature**](https://github.com/NIHAL-PM/theaibasedspeechdetrimentcalculator/issues)

  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge" />
    <img src="https://img.shields.io/badge/FastAPI-0.100%2B-00a393?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI Badge" />
    <img src="https://img.shields.io/badge/TensorFlow-2.0%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow Badge" />
    <img src="https://img.shields.io/badge/Tailwind_CSS-3.0-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS Badge" />
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License Badge" />
  </p>
</div>

---

## 📖 Table of Contents

- [🎯 About the Project](#-about-the-project)
- [✨ Key Features](#-key-features)
- [🧠 Architecture & Pipeline Mechanics](#-architecture--pipeline-mechanics)
- [📊 Dataset & Training Details](#-dataset--training-details)
- [🗂️ Project Structure](#️-project-structure)
- [🚀 Getting Started](#-getting-started)
- [💻 Usage](#-usage)
- [⚙️ Configuration](#-configuration)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 About the Project

The **Stuttering Detection AI Web App** is an interactive, browser-based speech therapy assistant. It captures live audio via your browser, slices the stream into continuous $3$-second blocks, and dynamically predicts stuttering events using a high-fidelity hybrid Deep Learning classifier trained on the **SEP-28k** dataset. When stuttering is detected above a specified threshold, the app automatically serves constructive, real-time therapy feedback.


```

```
   [Speak]             [3s Audio Webm]             [Inference Engine]

```

🎤 User Voice  ───►  📦 Web Audio API   ───►   🧠 FastAPI + CNN-LSTM Model
│
▼
🤖 Dynamic Feedback

```

> [!NOTE]
> This application runs completely on your local machine, keeping user voice recording data entirely private and edge-processed.

---

## ✨ Key Features

- **🔴 Continuous Browser Recording:** Uses the HTML5 `MediaRecorder` API to seamlessly capture and segment microphone input in real-time.
- **⚡ 120-Dimension Acoustic Fingerprint:** Captures vocal tract shape, velocity, and acceleration by extracting 40 Base MFCCs, 40 Deltas, and 40 Delta-Deltas.
- **💻 Edge-Optimized CPU Execution:** Engineered with structured batch normalization to perform fast neural network inference on consumer-grade hardware. No high-end GPU required.
- **🎨 Visual Status & Metrics UI:** Elegant user interface built with Tailwind CSS, animated status indicators, and color-coded results.
- **🎯 Dynamic Therapeutic Prompts:** Provides instant clinical recommendations (e.g., *Easy Onset*, *Slow Down*) to guide active speech adjustments.

---

## 🧠 Architecture & Pipeline Mechanics

### 1. Audio Capture & Normalization
The browser records voice inputs as a series of 3-second `.webm` files. When received by the backend, the raw signal is loaded, resampled to $16\text{ kHz}$ mono, and undergoes peak amplitude normalization:

$$x_{\text{norm}}[n] = \frac{x[n]}{\max_{k} |x[k]|}$$

This ensures volume fluctuations (due to mic gain differences) are mitigated before modeling.

### 2. Feature Extraction
The normalized audio data is translated from the time domain to the frequency domain, extracting a comprehensive acoustic matrix:


```

┌──────────────────────────────────────────────────────────┐
│              120 Acoustic Feature Matrix                 │
├─────────────────────────┬────────────────────────────────┤
│ 40 Base MFCCs           │ Captures vocal tract geometry  │
├─────────────────────────┼────────────────────────────────┤
│ 40 Delta MFCCs          │ Velocity of frame transitions  │
├─────────────────────────┼────────────────────────────────┤
│ 40 Delta-Delta MFCCs    │ Acceleration of audio dynamic  │
└─────────────────────────┴────────────────────────────────┘

```

The resulting sequence matrix is shaped to $(130, 120)$ and packaged as a $(1, 130, 120)$ tensor to match the input specifications of the trained model.

### 3. Classification & Decision Boundary
The input tensor flows through a hybrid 1D-Convolution and Recurrent Long Short-Term Memory (CNN-LSTM) network. The activation layer yields a prediction probability score $p \in [0.0, 1.0]$.

$$\text{Decision} = \begin{cases} \text{Stuttering Detected} & \text{if } p \ge \text{Threshold} \\ \text{Fluent} & \text{if } p < \text{Threshold} \end{cases}$$

> [!IMPORTANT]
> The default decision boundary is established at **$\text{Threshold} = 0.70$**. This ensures high specificity and protects against background noise artifacts.

---

## 📊 Dataset & Training Details

The core `.h5` model is trained on the curated **SEP-28k Dataset** (comprising annotations for stuttering events in public podcasts).

* **Download Constraints Handled:** Cleaned and accounted for invalid/dead podcast URLs in the original metadata file.
* **Volume:** $20,906$ valid structural audio clips generated.
* **Class Balances:**
  - **Fluent Samples:** $11,537$
  - **Stuttering Samples:** $9,369$
* **Neural Architecture:** A Keras sequential setup featuring:
  - `Conv1D` and `MaxPooling1D` for spatial spectrogram features.
  - Double `LSTM` layers to capture speech flow over time.
  - `BatchNormalization` and `Dropout` layers to prevent overfitting.
  - Training accelerated with dynamic learning rate decay (`ReduceLROnPlateau`) and `class_weight` adjustments to mitigate class imbalance.

---

## 🗂️ Project Structure

| File / Folder | Role in Pipeline |
| :--- | :--- |
| **`app.py`** | Core engine containing the FastAPI server and the embedded HTML/JS frontend UI. |
| **`stutter_detector_model.h5`** | The trained weights and structural configuration of the CNN-LSTM neural net. |
| **`requirements.txt`** | Dependency definitions file to build the environmental workspace. |
| **`README.md`** | Comprehensive developer and project deployment documentation. |

---

## 🚀 Getting Started

Follow these steps to configure and run the software locally.

### Prerequisites

- **Python:** Version `3.8` to `3.12` recommended.
- **Pip:** Standard Python package manager.
- **Model File:** The pre-trained file `stutter_detector_model.h5` must be placed in the root directory alongside `app.py`.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/yourusername/stuttering-detection-ai.git](https://github.com/yourusername/stuttering-detection-ai.git)
   cd stuttering-detection-ai

```

2. **Initialize a Virtual Environment:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

```


3. **Install Requirements:**
```bash
pip install fastapi uvicorn python-multipart librosa tensorflow numpy

```



---

## 💻 Usage

To run the application, follow these steps:

1. **Start the FastAPI / Uvicorn Server:**
```bash
python app.py

```


2. **Access the Interface:**
Launch your preferred modern web browser and navigate to:
```text
[http://127.0.0.1:8000](http://127.0.0.1:8000)

```


3. **In-App Navigation:**
* Press Start Session to unlock browser recording.
* Speak naturally into the device microphone.
* Watch the visual status bars update dynamically every $3$ seconds based on your speech characteristics.



---

## ⚙️ Configuration

You can tweak sensitivity settings directly inside **`app.py`**:

```python
MODEL_PATH = "stutter_detector_model.h5"  # Loaded model path
SAMPLE_RATE = 16000                       # Audio sampling frequency
DURATION = 3                              # Continuous segment length (sec)
THRESHOLD = 0.70                          # Adjust sensitivity (0.0 to 1.0)

```

---

## 🛠️ Troubleshooting

* **`ModuleNotFoundError: No module named 'fastapi'`**:
Make sure your virtual environment is active before running installation commands.
* **`CUDA Error / No CUDA Drivers` Warning**:
You can safely ignore this! The script will fallback automatically to your system CPU without any performance degradation for single-stream predictions.
* **Microphone Access Blocked**:
Your browser requires a secure server context to process audio. Access the site via `http://localhost:8000` or `http://127.0.0.1:8000`. Running on standard IP layouts (e.g., raw networking protocols) without SSL will trigger browser security blocks.

---

## 🤝 Contributing

We welcome community collaborations!

1. Fork this repository.
2. Form your working feature directory (`git checkout -b feature/NewTherapyMetrics`).
3. Commit structural updates (`git commit -m 'Added dynamic confidence bar charts'`).
4. Push to remote (`git push origin feature/NewTherapyMetrics`).
5. Open a Pull Request for code review.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgments

* **Apple Machine Learning Research:** For publishing the invaluable [SEP-28k Dataset](https://github.com/apple/ml-stuttering-events-dataset).
* **The Librosa Development Team:** For engineering standard python acoustics processing workflows.
* **FastAPI & Uvicorn Developers:** For making local API service high-speed and straightforward.

