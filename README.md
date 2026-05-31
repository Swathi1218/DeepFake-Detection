# Deepfake Video Detection using ResNeXt + BiLSTM

A deep learning pipeline for detecting deepfake videos using a **ResNeXt50 CNN** for per-frame spatial feature extraction combined with a **Bidirectional LSTM** for temporal sequence modelling, trained on the **Celeb-DF v2** dataset.

---

## Project Overview

Deepfake videos are synthetically generated face-swapped videos increasingly difficult to detect with the naked eye. This project builds a binary video classifier (real vs. fake) by:

1. Sampling 16 frames uniformly from each video
2. Extracting deep spatial features per frame using a pretrained ResNeXt50 CNN
3. Modelling temporal dependencies across frames with a Bidirectional LSTM
4. Classifying the video as **real** or **fake**

---

## Model Architecture

```
Input Video
    │
    ▼
Frame Sampling (16 frames)         ← uniform temporal sampling
    │
    ▼
ResNeXt50-32x4d (pretrained)       ← CNN backbone, FC layer removed
Output: 2048-dim feature per frame    ImageNet pretrained weights, frozen
    │  [B, 16, 2048]
    ▼
Bidirectional LSTM                 ← hidden_size=256, num_layers=1
Output: 512-dim (256 × 2)             models temporal patterns
    │  last hidden state
    ▼
Linear Classifier                  ← 512 → 2 (real / fake)
    │
    ▼
  real / fake
```

**Key design decisions:**
- ResNeXt backbone is **frozen** — only LSTM and classifier layers are trained
- **WeightedRandomSampler** to handle class imbalance between real and fake videos
- **Bidirectional** LSTM so each frame benefits from both past and future context

---

## Repository Structure

```
deepfake-detection/
│
├── DEEP_FAKE_DETECTION.ipynb     ← main notebook (data loading, training, evaluation)
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md                 ← how to download Celeb-DF v2
│
├── models/
│   └── README.md                 ← info about saved model weights
│
└── assets/                       ← confusion matrix, ROC curve images
```

---

## Dataset

This project uses the **[Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)** dataset.

| Split | Proportion |
|-------|-----------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Stratified splitting ensures class balance across all splits. See [`data/README.md`](data/README.md) for download instructions.

---

## How to Run

### On Kaggle (Recommended — dataset is natively available)

1. Open `DEEP_FAKE_DETECTION.ipynb` in a Kaggle notebook
2. Add the **Celeb-DF v2** dataset:
   - Go to **Add Data → Search "Celeb-DF v2"** and attach it
3. Enable GPU: **Settings → Accelerator → GPU T4**
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

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct classifications |
| Precision | Of predicted fakes, how many are truly fake |
| Recall | Of actual fakes, how many were caught |
| Specificity | Of actual reals, how many were correctly identified |
| F1 Score | Harmonic mean of precision and recall |
| AUC-ROC | Area under ROC curve |
| EER | Equal Error Rate (where FPR = FNR) |



## Technical Notes

- **Frame sampling:** 16 frames uniformly sampled per video. Videos shorter than 16 frames are tiled to fill the sequence.
- **Transforms:** Resize to 224×224 + ImageNet normalization.
- **Class imbalance:** Handled via `WeightedRandomSampler` during training.
- **Optimizer:** Adam, lr=1e-4, up to 20 epochs
- **Best model** saved automatically based on highest validation accuracy → `best_model.pth`
