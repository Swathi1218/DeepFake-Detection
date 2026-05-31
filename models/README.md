# Dataset

This project uses the **Celeb-DF v2 (Celebrity Deepfake)** dataset.

## About the Dataset

Celeb-DF v2 contains high-quality deepfake videos synthesized from celebrity interviews sourced from YouTube. It is one of the most challenging deepfake benchmarks due to its improved visual quality compared to older datasets.

| Category | Description |
|----------|-------------|
| `Celeb-real` | Real celebrity interview videos |
| `Celeb-synthesis` | Deepfake versions of those videos |

## How to Download

### Option A — Kaggle (Recommended)

The dataset is available directly on Kaggle:

1. Go to: https://www.kaggle.com/datasets/reubensuju/celeb-df-v2
2. Click **Download**
3. Place and unzip under `data/`:

```
data/
└── celeb-df-v2/
    ├── Celeb-real/
    │   ├── id0_0000.mp4
    │   ├── id0_0001.mp4
    │   └── ...
    └── Celeb-synthesis/
        ├── id0_id1_0000.mp4
        ├── id0_id1_0001.mp4
        └── ...
```

### Option B — Official Source

1. Request access at the official repo: https://github.com/yuezunli/celeb-deepfakeforensics
2. Fill in the Google Form linked in the repo
3. You will receive a download link via email

## Note

The dataset files are excluded from this repository via `.gitignore` due to their large size. Do **not** commit video files.
