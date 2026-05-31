# 🎭 Deepfake Video Detection using ResNeXt + BiLSTM

A deep learning pipeline for detecting deepfake videos using a **ResNeXt50 CNN** for per-frame spatial feature extraction combined with a **Bidirectional LSTM** for temporal sequence modelling, trained on the **Celeb-DF v2** dataset.

---

## 📌 Project Overview

Deepfake videos are synthetically generated face-swapped videos that are increasingly difficult to detect with the naked eye. This project builds a binary video classifier (real vs. fake) by:

1. Sampling frames uniformly from each video
2. Extracting deep spatial features per-frame using a pretrained ResNeXt50
3. Modelling temporal dependencies across frames with a Bidirectional LSTM
4. Classifying the video as **real** or **fake**

---

## 🏗️ Model Architecture

```
Input Video
    │
    ▼
┌─────────────────────────────────┐
│   Frame Sampling (16 frames)    │  ← uniform temporal sampling
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│   ResNeXt50-32x4d (pretrained)  │  ← CNN backbone, fc layer removed
│   Output: 2048-dim feature/frame│    ImageNet pretrained weights
└─────────────────────────────────┘
    │  [B, 16, 2048]
    ▼
┌─────────────────────────────────┐
│   Bidirectional LSTM            │  ← hidden_size=256, num_layers=1
│   Output: 512-dim (256 × 2)     │    models temporal patterns
└─────────────────────────────────┘
    │  last hidden state
    ▼
┌─────────────────────────────────┐
│   Linear Classifier             │  ← 512 → 2 (real / fake)
└─────────────────────────────────┘
    │
    ▼
  real / fake
```

**Key design decisions:**
- ResNeXt backbone is **frozen** — only LSTM and classifier are trained
- **WeightedRandomSampler** to handle class imbalance between real and fake videos
- **Bidirectional** LSTM so each frame benefits from both past and future context

---

## 📂 Repository Structure

```
deepfake-detection/
│
├── DEEP_FAKE_DETECTION.ipynb     ← main notebook (training + evaluation)
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── dataset.py                ← VideoDataset class
│   ├── model.py                  ← ResNeXtLSTM model definition
│   ├── train.py                  ← training and evaluation loop functions
│   └── metrics.py                ← AUC, EER, confusion matrix utilities
│
├── data/
│   └── README.md                 ← how to download Celeb-DF v2
│
├── models/
│   └── README.md                 ← info about saved model weights
│
└── assets/
    └── README.md                 ← place confusion matrix / ROC curve images here
```

---

## 📊 Dataset

This project uses the **[Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)** dataset.

| Split | Proportion |
|-------|-----------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Stratified splitting ensures class balance across all splits. See [`data/README.md`](data/README.md) for download instructions.

---

## 🚀 How to Run

### On Kaggle (Recommended — dataset is available natively)

1. Open `DEEP_FAKE_DETECTION.ipynb` in a Kaggle notebook
2. Add the **Celeb-DF v2** dataset from Kaggle Datasets
3. Enable GPU: **Settings → Accelerator → GPU**
4. Run all cells: **Run All**

### Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-detection.git
   cd deepfake-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (see [`data/README.md`](data/README.md)) and update the path in the notebook:
   ```python
   root_dir = Path("/your/local/path/to/celeb-df-v2")
   ```

4. Launch Jupyter and run:
   ```bash
   jupyter notebook DEEP_FAKE_DETECTION.ipynb
   ```

---

## 📈 Evaluation Metrics

The model is evaluated using:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct classifications |
| Precision | Of predicted fakes, how many are truly fake |
| Recall | Of actual fakes, how many were caught |
| Specificity | Of actual reals, how many were correctly identified |
| F1 Score | Harmonic mean of precision and recall |
| AUC-ROC | Area under ROC curve |
| EER | Equal Error Rate (where FPR = FNR) |

---

## 🛠️ Requirements

- Python 3.8+
- CUDA-compatible GPU (strongly recommended)
- PyTorch 2.x
- See `requirements.txt` for full list

---

## 📁 Saved Model

The best model weights (`best_model.pth`) are saved during training based on highest validation accuracy. See [`models/README.md`](models/README.md).

---

## 🔬 Technical Notes

- **Frame sampling:** 16 frames are uniformly sampled per video. If a video has fewer than 16 frames, they are tiled/repeated to fill the sequence.
- **Augmentation:** Resize to 224×224, ImageNet normalization (mean/std).
- **Class imbalance:** Celeb-DF v2 has more fake than real videos. Handled via `WeightedRandomSampler` during training.
- **Optimizer:** Adam, lr=1e-4
- **Loss:** CrossEntropyLoss
- **Training:** Up to 20 epochs; best checkpoint saved based on val accuracy
