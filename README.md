# ISL Continuous Sign Language Translator

Continuous Indian Sign Language (ISL) recognition system using a hybrid ST-GCN + Transformer architecture. Trained and evaluated on the [INCLUDE dataset](https://zenodo.org/records/4010759) (263 word-level signs, 4,287 videos).

**Best result: 30.77% WER** on word-level continuous recognition.

## Architecture

1. **Keypoint extraction** - MediaPipe Holistic extracts 543 2D keypoints per frame (33 pose, 468 face, 21×2 hand).
2. **ST-GCN encoder** - Models spatial relationships across the skeletal graph structure.
3. **Transformer encoder** - Captures temporal dependencies across the frame sequence.
4. **CTC head** - Connectionist Temporal Classification for alignment-free gloss prediction.
5. **IndicTrans2 (optional)** - Translates predicted gloss sequences into Hindi/English text.

## Repository Structure

```
isl-translator/
├── src/
│   ├── preprocessing/       # Keypoint extraction, dataset loaders
│   ├── models/              # ST-GCN, Transformer, CTC decoder
│   └── training/            # Trainer, mixed-precision, metrics
├── notebooks/               # Kaggle training notebooks
├── scripts/                 # CLI entry points
└── configs/                 # Hyperparameter configs (YAML)
```

## Reproducibility

Training and keypoint extraction were run on Kaggle due to dataset size and compute requirements. Only the primary notebooks are included here; additional resumption notebooks (needed due to Kaggle session timeouts) follow the same setup.

- `01_extract_keypoints.ipynb` - MediaPipe keypoint extraction from raw video.
- `02_pretraining_include.ipynb` - Full training pipeline on INCLUDE.
- `03_finetune_csltr.ipynb` - Fine-tuning for sentence-level translation.

## Local Inference

Requires an NVIDIA GPU with ≥8 GB VRAM (tested on RTX 4060).

```bash
git clone https://github.com/Sid-V5/isl-translator.git
cd isl-translator
pip install -r requirements.txt

# Run the webcam demo
python src/inference/demo.py --checkpoint checkpoints/best.pt
```

## Configuration

All hyperparameters are in `configs/config.yaml`. Training used cosine-annealed LR (down to 1e-6) with label smoothing disabled.

## License

MIT
