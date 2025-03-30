# Sign Language Detection Project

This project implements a sign language detection pipeline using video inputs and deep learning models. It processes video data through several stages including preprocessing, feature extraction, transformer-based recognition, and optional segmentation to improve detection performance.

## Project Structure

```
sign-language-detection/
├── README.md                       # Project documentation
├── data/                           # Data storage
│   ├── videos/                     # Original video files
│   ├── frames/                     # Extracted frames from videos
│   ├── features/                   # Extracted features
│   └── labels/                     # Label files (CSV, pkl)
├── models/                         # Model weights and checkpoints
├── src/                            # Source code
│   ├── preprocessing/              # Preprocessing scripts
│   │   ├── video_to_frames.py      # Video to frame extraction
│   │   └── feature_extraction.py   # Feature extraction utilities
│   ├── models/                     # Model definitions
│   │   ├── transformer.py          # Gesture transformer implementation
│   │   └── vit.py                  # Vision transformer implementation
│   ├── segmentation/               # Segmentation utilities
│   │   └── segment.py              # Segmentation implementation
│   ├── inference/                  # Inference scripts
│   │   └── predict.py              # Prediction utilities
│   └── utils/                      # Utility functions
│       └── data_loaders.py         # Data loading utilities
├── scripts/                        # Pipeline scripts
│   ├── run_pipeline.sh             # Main pipeline execution script
│   └── slurm_job.sh                # SLURM cluster script
├── tests/                          # Test cases
└── logs/                           # Log files
```

## Pipeline Overview

1. **Preprocessing:**
   - Convert input videos (`.webm`) to image frames using `src/preprocessing/video_to_frames.py`.
   - Extract features from frames using a deep model (e.g., ResNet or Vision Transformer).

2. **Feature Loading & Preparation:**
   - Load the saved features using utilities in `src/utils/data_loaders.py`.
   - Map features to the corresponding labels using data from `data/labels/`.

3. **Model Inference:**
   - Run inference using scripts in `src/inference/`, which apply the transformer models on the features.

4. **Segmentation (Optional):**
   - The segmentation module (`src/segmentation/segment.py`) performs segmentation on the input frames to potentially enhance recognition accuracy.

5. **Execution:**
   - Use `scripts/run_pipeline.sh` for running the complete pipeline locally.
   - Use `scripts/slurm_job.sh` if deploying the pipeline on a SLURM-managed cluster.

## Getting Started

### Prerequisites

- Python 3.8 or above
- PyTorch
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sign-language-detection.git
   cd sign-language-detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. **Video to Frames Conversion:**
   ```
   python -m src.preprocessing.video_to_frames --input data/videos/example.webm --output data/frames/example/
   ```

2. **Feature Extraction:**
   ```
   python -m src.preprocessing.feature_extraction --input data/frames/example/ --output data/features/example_features.pkl
   ```

3. **Inference:**
   ```
   python -m src.inference.predict --features data/features/example_features.pkl --model models/gesture_transformer.pth
   ```

4. **Run Full Pipeline:**
   ```
   bash scripts/run_pipeline.sh data/videos/example.webm
   ```

## Notes

- Ensure that the label mapping file is correctly aligned with your training data.
- Check the log files in the `logs/` directory for detailed run-time information.
- For advanced segmentation, the project can integrate with the Sapiens framework for human-centric vision tasks.

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgements

This project integrates parts of the Sapiens framework for human-centric vision tasks. Special thanks to the developers behind Sapiens for their open-source contributions. 