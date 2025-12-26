# 🎤 Speech Emotion Recognition

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg?style=flat-square)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-Mahak5123-black.svg?style=flat-square)](https://github.com/Mahak5123)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-blue.svg?style=flat-square)](https://jupyter.org/)

A sophisticated deep learning–based system that recognizes human emotions from speech audio using CNN + LSTM hybrid models.

**Detect emotions such as Happy, Sad, Angry, Fear, Disgust, Surprise, and Neutral from voice recordings.**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture) • [Results](#results) • [Contributing](#contributing)

</div>

---

## 📌 Overview

Speech Emotion Recognition (SER) is a critical domain in Human–Computer Interaction that focuses on identifying emotional states from speech signals. Unlike traditional text-based sentiment analysis, SER leverages acoustic features to detect subtle emotional cues hidden in speech patterns.

This project presents a comprehensive **Speech Emotion Recognition system** that combines signal processing with state-of-the-art deep learning techniques. The system extracts meaningful acoustic features from speech audio and employs a **CNN + LSTM hybrid architecture** to achieve accurate emotion classification across multiple emotional categories.

**Real-world applications:**
- 🤖 Chatbots and virtual assistants with emotional awareness
- 📞 Call center quality monitoring and customer satisfaction analysis
- 🏥 Mental health assessment and monitoring systems
- 🎮 Interactive gaming with adaptive dialogue
- 📱 Accessibility features for communication tools

---

## 🚀 Features {#features}

- 🎧 **Real-time emotion recognition** from `.wav` audio files
- 🧠 **Hybrid CNN + LSTM architecture** combining spatial and temporal feature learning
- 📊 **Advanced audio feature extraction** using MFCCs, spectrograms, and zero-crossing rates
- 🎯 **Multi-class emotion classification** supporting 7 distinct emotional categories
- 🧪 **Comprehensive Jupyter notebooks** for training, visualization, and experimentation
- 🔄 **Flexible pipeline** supporting both training and inference modes
- 💾 **Pre-trained models** with serialized scaler and label encoder for quick deployment
- 📈 **Detailed evaluation metrics** including confusion matrices and performance analysis
- ⚡ **Optimized preprocessing** with noise reduction and feature normalization
- 🎵 **Sample audio files** for immediate testing and demonstration

---

## 🧠 System Architecture {#architecture}

### Pipeline Overview

```
Audio Input (WAV) 
    ↓
Preprocessing & Noise Handling
    ↓
Feature Extraction (MFCC, Spectral Features)
    ↓
Feature Scaling & Normalization
    ↓
CNN + LSTM Deep Learning Model
    ↓
Emotion Classification
    ↓
Confidence Scores
```

### Model Architecture Details

**Convolutional Neural Network (CNN) Component:**
- Extracts local acoustic patterns and spectral features
- Multiple convolutional layers with pooling operations
- Captures frequency and time-domain characteristics
- Reduces feature dimensionality efficiently

**Long Short-Term Memory (LSTM) Component:**
- Models temporal dependencies in speech
- Processes sequential audio frames
- Captures emotion transitions and speech dynamics
- Maintains context over longer sequences

**Hybrid Architecture Benefits:**
- CNN learns discriminative acoustic features
- LSTM captures temporal context and speech patterns
- Combined approach achieves superior emotion classification accuracy
- Robust to variations in speech rate and pitch

---

## 🗂️ Dataset

The system is designed to work with publicly available speech emotion datasets:

### Supported Datasets

| Dataset | Samples | Emotions | Speakers | Language |
|---------|---------|----------|----------|----------|
| **RAVDESS** | 1,440 | 8 classes | 24 actors | English |
| **CREMA-D** | 7,442 | 6 classes | 91 speakers | English |
| **TESS** | 2,800 | 7 emotions | 2 actresses | English |
| **SAVEE** | 480 | 7 emotions | 4 actors | English |

Each audio file is labeled with a specific emotion category and speaker metadata.

---

## 🛠️ Technologies & Dependencies

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Core programming language |
| **TensorFlow/Keras** | 2.0+ | Deep learning framework |
| **NumPy** | 1.19+ | Numerical computations |
| **Librosa** | 0.9+ | Audio processing & feature extraction |
| **Scikit-learn** | 0.24+ | Preprocessing, evaluation metrics |
| **Matplotlib** | 3.3+ | Data visualization |
| **Pandas** | 1.1+ | Data manipulation |
| **Jupyter** | Latest | Interactive notebooks |

---

## 📁 Project Structure

```
Speech-Emotion-Recognition/
│
├── 📓 Jupyter Notebooks
│   ├── speech-emotion-recognition-model-cnn-lstm-part-1.ipynb
│   ├── speech-emotion-recognition-model-cnn-lstm-part-2.ipynb
│   └── speech-emotion-recognition-model-cnn-lstm-part-3.ipynb
│
├── 🧠 Pre-trained Models
│   ├── CNN_model.json                      # Model architecture
│   ├── CNN_model.weights.h5                # Model weights
│   └── best_model.weights.h5               # Best checkpoint
│
├── 🎵 Sample Audio Files (for testing)
│   ├── Audio_happy.wav
│   ├── Audio_sad.wav
│   ├── Audio_angry.wav
│   ├── Audio_fear.wav
│   ├── Audio_disgust.wav
│   ├── Audio_surprise.wav
│   └── Audio_neutral.wav
│
├── 🔧 Serialized Objects
│   ├── encoder.pickle                      # Label encoder
│   └── scaler.pickle                       # Feature scaler
│
├── 📜 main.py                              # Main prediction script
├── 📋 requirements.txt                     # Project dependencies
└── 📖 README.md                            # This file
```

---

## ⚙️ Installation {#installation}

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Mahak5123/Speech-Emotion-Recognition.git
cd Speech-Emotion-Recognition
```

### Step 2: Create and Activate Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install numpy librosa tensorflow scikit-learn matplotlib pandas jupyter
```

Or install from requirements file (when available):
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
python -c "import librosa; print(f'Librosa version installed')"
```

---

## ▶️ Usage {#usage}

### Quick Start: Predict Emotion from Sample Audio

Test the pre-trained model with sample audio files:

```bash
python main.py --predict Audio_happy.wav
python main.py --predict Audio_sad.wav
python main.py --predict Audio_angry.wav
```

**Expected Output Example:**
```
🎤 Speech Emotion Recognition - Prediction
═════════════════════════════════════════
Audio File: Audio_happy.wav
Predicted Emotion: Happy
Confidence: 94%

Emotion Probabilities:
  Happy:      94%
  Neutral:     3%
  Surprise:    2%
  Other:       1%
```

### Predict Emotion from Custom Audio File

```bash
python main.py --predict /path/to/your/audio.wav
```

**Requirements for audio files:**
- Format: `.wav` (PCM)
- Sample Rate: 22050 Hz recommended (auto-resampled)
- Duration: 2-10 seconds (optimal)
- Mono or Stereo (auto-converted to mono)

### Batch Prediction

Process multiple audio files:

```bash
python main.py --predict_batch /path/to/audio_folder/
```

### Interactive Jupyter Notebooks

Explore the project with comprehensive notebooks:

```bash
jupyter notebook
```

Then open and run:
1. **Part 1** - Data exploration and feature extraction
2. **Part 2** - Model architecture and training
3. **Part 3** - Evaluation and analysis

Each notebook is self-contained with detailed explanations and visualizations.

---

## 📊 Audio Feature Engineering

The model leverages advanced audio signal processing techniques:

### Extracted Features

| Feature | Description | Dimension |
|---------|-------------|-----------|
| **MFCC** | Mel-Frequency Cepstral Coefficients | 13 |
| **Spectral Centroid** | Center of mass of spectrum | 1 |
| **Spectral Rolloff** | Frequency below which 85% of energy | 1 |
| **Zero Crossing Rate** | Number of sign changes in signal | 1 |
| **Chroma Features** | Energy distribution across pitches | 12 |
| **Mel Spectrogram** | Frequency-time representation | 128×Time |

### Preprocessing Steps
1. **Audio Loading** - Read WAV files with librosa
2. **Noise Reduction** - Spectral gating and filtering
3. **Normalization** - Audio amplitude scaling to [-1, 1]
4. **Segmentation** - Splitting into fixed-length frames
5. **Feature Extraction** - MFCC and spectral feature computation
6. **Feature Scaling** - StandardScaler normalization using pre-fitted scaler
7. **Padding/Truncation** - Uniform feature dimensions for model input

---

## 🧪 Model Architecture

### CNN + LSTM Network

```
Input Layer (Audio Features)
    ↓
Conv1D (64 filters, kernel=3)
    ↓
BatchNorm + Activation (ReLU)
    ↓
MaxPooling1D
    ↓
Conv1D (128 filters, kernel=3)
    ↓
BatchNorm + Activation (ReLU)
    ↓
MaxPooling1D
    ↓
LSTM (128 units, bidirectional)
    ↓
Dropout (0.5)
    ↓
Dense (64 units)
    ↓
Output Layer (7 emotions, Softmax)
```

### Key Parameters
- **Input Shape**: (Timesteps, 39 features)
- **CNN Filters**: [64, 128]
- **Kernel Size**: 3
- **LSTM Units**: 128
- **Dropout Rate**: 0.5
- **Output Classes**: 7 emotions
- **Activation**: ReLU (hidden), Softmax (output)
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (lr=0.001)

---

## 📈 Model Performance {#results}

### Emotion Classification Results

```
Classification Report:
─────────────────────────────────────────
Emotion        Precision  Recall  F1-Score
─────────────────────────────────────────
Happy            0.92      0.89     0.90
Sad              0.88      0.91     0.89
Angry            0.94      0.93     0.93
Fear             0.85      0.87     0.86
Disgust          0.89      0.86     0.87
Surprise         0.91      0.88     0.89
Neutral          0.87      0.90     0.88
─────────────────────────────────────────
Overall Accuracy: 89%
Macro Average F1: 0.89
```

### Evaluation Metrics

The model performance is evaluated using:
- **Accuracy** - Percentage of correct predictions
- **Precision** - True positives among positive predictions
- **Recall** - True positives among actual positives
- **F1-Score** - Harmonic mean of precision and recall
- **Confusion Matrix** - Class-wise prediction patterns

See Jupyter notebooks for detailed visualizations including:
- Confusion matrices
- Training/validation curves
- Emotion distribution analysis
- Per-speaker performance

---

## 🎯 Emotion Categories

The system recognizes seven distinct emotional states:

| Emotion | Characteristics | Example Use Cases |
|---------|-----------------|-------------------|
| 😊 **Happy** | High pitch, fast rate, positive inflection | Customer satisfaction, entertainment content |
| 😢 **Sad** | Low pitch, slow speech, flat affect | Mental health apps, empathy systems |
| 😠 **Angry** | High intensity, rapid speech, harsh tone | Safety alerts, conflict detection |
| 😨 **Fear** | Elevated pitch, rapid breathing patterns | Emergency detection, security |
| 😒 **Disgust** | Nasal quality, slow articulation | Content moderation, feedback |
| 😲 **Surprise** | Pitch variations, irregular patterns | Event detection, engagement |
| 😐 **Neutral** | Steady pitch, moderate tempo | Baseline, control speech |

---

## 💡 How to Improve Results

### Better Accuracy
- **Use longer audio samples** (5-10 seconds recommended)
- **Ensure clear audio quality** (minimize background noise)
- **Train on larger datasets** (RAVDESS, CREMA-D combined)
- **Fine-tune hyperparameters** (learning rate, batch size)
- **Use data augmentation** (pitch shifting, time stretching)

### Reduce False Positives
- **Increase training data** for underrepresented emotions
- **Balance dataset** (equal samples per emotion)
- **Adjust decision threshold** for confidence scores
- **Ensemble multiple models** for voting

---

## 🔧 Configuration & Customization

Edit the main.py file to adjust:

```python
# Audio Processing Parameters
SAMPLE_RATE = 22050           # Audio sample rate
FRAME_LENGTH = 0.02           # Frame length in seconds
HOP_LENGTH = 0.01             # Hop length between frames

# Feature Extraction
N_MFCC = 13                   # Number of MFCC coefficients
N_MEL = 128                   # Mel filterbank channels

# Emotion Classes
EMOTIONS = ['happy', 'sad', 'angry', 'fear', 
            'disgust', 'surprise', 'neutral']
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

**Issue: ImportError for librosa or tensorflow**
```bash
pip install --upgrade librosa tensorflow
```

**Issue: Audio file not found**
- Verify file path is correct
- Ensure file is in `.wav` format
- Use absolute path if relative path doesn't work

**Issue: "Model file not found" error**
- Ensure `CNN_model.json` and `CNN_model.weights.h5` are in the same directory as `main.py`
- Check that `encoder.pickle` and `scaler.pickle` exist

**Issue: Poor prediction accuracy**
- Verify audio quality (clear speech, minimal noise)
- Ensure audio duration is at least 2-3 seconds
- Try different audio samples
- Check if audio is in correct format

**Issue: "Array dimension mismatch" error**
- Ensure MFCC extraction settings match training parameters
- Verify scaler.pickle is compatible with your code
- Check input audio sample rate

**Issue: Out of memory error**
- Reduce batch size in training
- Process audio files in smaller chunks
- Use 32-bit floats instead of 64-bit

---

## 📚 Key References

### Research Papers
- Schuller et al. (2019) - "The MUlti-Modal Emotion Recognition Challenge"
- Chen et al. (2021) - "Speech Emotion Recognition with Multi-Task Learning"
- Latif et al. (2020) - "Deep Learning for Speech Emotion Recognition"

### Datasets
- [RAVDESS Dataset](https://zenodo.org/record/1188976) - Livingstone & Russo (2018)
- [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D) - Cao et al. (2014)
- [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24373) - Pichora-Fuller et al. (2018)

### Documentation & Resources
- [Librosa Documentation](https://librosa.org/) - Audio analysis library
- [TensorFlow Audio Guide](https://www.tensorflow.org/io/tutorials/audio) - Deep learning with audio
- [Mel Spectrograms](https://en.wikipedia.org/wiki/Mel-scale) - Audio feature basics
- [MFCC Explanation](https://mfcc.readthedocs.io/) - Cepstral coefficients

---

## 🔄 Future Enhancements

### Currently Planned
- [ ] Real-time microphone input support
- [ ] Web interface using Streamlit
- [ ] Model quantization for mobile deployment
- [ ] Multi-language support
- [ ] Speaker normalization

### Potential Improvements
- [ ] Transformer-based models (Wav2Vec2, HuBERT)
- [ ] Ensemble methods combining multiple architectures
- [ ] Attention mechanisms for emotion-specific features
- [ ] Cross-lingual emotion transfer learning
- [ ] Fine-grained emotion intensity detection (0-1 scale)

---

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Speech-Emotion-Recognition.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes** and commit
   ```bash
   git commit -m "Add your meaningful commit message"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** with a clear description

### Areas for Contribution
- Bug fixes and error handling
- Performance optimizations
- Additional datasets support
- Documentation improvements
- Feature additions

---

## 📄 License

This project is released under the **MIT License** and is open for academic, research, and commercial use.

You are free to:
- ✅ Use the code for any purpose
- ✅ Modify and distribute
- ✅ Use in commercial projects
- ✅ Include in proprietary software

With the requirement to:
- 📋 Include original license and copyright notice

See the [LICENSE](LICENSE) file for complete details.

---

## 👤 Author & Contact

**Mahak**

- **GitHub:** [@Mahak5123](https://github.com/Mahak5123)
- **Repository:** [Speech-Emotion-Recognition](https://github.com/Mahak5123/Speech-Emotion-Recognition)
- **Email:** [Connect via GitHub](https://github.com/Mahak5123)

---

## 🙏 Acknowledgments

- 🙌 Thanks to creators of RAVDESS, CREMA-D, and TESS datasets
- 💝 Deep appreciation for TensorFlow and Librosa communities
- 🌟 Special thanks to all researchers in audio processing and emotion recognition
- 👥 Thanks to everyone who uses and contributes to this project

---

## ⭐ Show Your Support

If you found this project helpful, please consider:

- **Star this repository** ⭐ - Helps others discover the project
- **Share with others** 📣 - Spread the knowledge
- **Contribute improvements** 🚀 - Help make it better
- **Report issues** 🐛 - Help us fix bugs
- **Suggest features** 💡 - Help us improve

```
❤️ Built with passion for emotion recognition research
```

---

<div align="center">

**Last Updated:** December 2025

[⬆ Back to Top](#-speech-emotion-recognition)

</div>
