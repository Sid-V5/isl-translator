# ISL Continuous Sign Language Translator: Achieving 30.77% WER

A state-of-the-art (SOTA) continuous Indian Sign Language (ISL) recognition and translation system utilizing a hybrid Spatio-Temporal Graph Convolutional Network (ST-GCN) and Transformer architecture. Optimized for real-time inference on edge GPUs (e.g., RTX 4060 8GB).

## 🚀 Key Results
*   **30.77% Word Error Rate (WER)** achieved on the expansive INCLUDE dataset (4,287 signs, word-level).
*   **Highly efficient pose-based processing** (MediaPipe Holistic) completely eliminating the need for raw video input during inference.
*   **SOTA-competitive** training architecture scaling from isolated signs to continuous sentence-level recognition.

## 🧠 Architecture Overview
This model leverages a hybrid spatial-temporal approach to capture the intricate nuances of Indian Sign Language:
1.  **MediaPipe Holistic Extraction:** Extracts 543 2D keypoints per frame (33 pose, 468 face, 21x2 hand).
2.  **ST-GCN Module:** Models spatial dependencies across the human skeletal graph.
3.  **Transformer Encoder:** Models long-range temporal dependencies across the sequence of frames.
4.  **CTC Decoder:** Connectionist Temporal Classification loss allows alignment-free training, translating frame representations directly to glosses.
5.  **IndicTrans2 (Optional):** Gloss sequence translation to natural Hindi/English text.

## 📂 Repository Structure
```
isl-translator/
├── src/                     # Core PyTorch Architecture
│   ├── preprocessing/       # Keypoint extraction, pyTorch datasets
│   ├── models/              # ST-GCN, Transformer, CTC Head
│   └── training/            # Custom Trainer, Mixed-Precision, Metrics
├── notebooks/               # Executed Colab/Kaggle Training Notebooks
├── scripts/                 # CLI entry points for training & data eval
└── configs/                 # YAML Hyperparameter tracking
```

## 🛠️ Reproducibility (Kaggle/Colab)
Due to dataset sizes and computation requirements, training and keypoint extraction were executed in Kaggle Notebook environments. *Note: For professional clarity, only the primary execution notebooks are provided here. Subsequent resumption notebooks (necessitated by Kaggle's session timeouts) follow the identical setup.*
*   `01_extract_keypoints.ipynb`: MediaPipe keypoint extraction methodology.
*   `02_pretraining_include.ipynb`: The training architecture and logic that culminated in the **30.77% WER**.
*   `03_finetune_csltr.ipynb`: Complete guide to fine-tuning the model for sentence-level translation.

## 💻 Local Setup (Inference)
The architecture is designed to run inference smoothly on an 8GB VRAM GPU.
```bash
# Clone the repository
git clone https://github.com/lastlegend/isl-translator.git
cd isl-translator

# Install dependencies via uv or pip
uv venv
uv pip install -r requirements.txt
```

## 📝 Configuration
Hyperparameters are managed via `configs/config.yaml`. Note that label smoothing was disabled and learning rates were cosine-annealed to 1e-6 to achieve the final SOTA results.

## 📄 License
This project is licensed under the MIT License.
