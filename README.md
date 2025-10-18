# IITP-Smarthome

## Environment Setup
1. `conda create -n smarthome python=3.10`
2. `conda activate smarthome`
3. `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
4. `python -m pip install numpy pandas neurokit2 fire tqdm scipy datasets scikit-learn`

## Repository Layout
- `wesad/` &mdash; preprocessing, data, model, trainer, and config utilities for the WESAD dataset.
- `scripts/train_wesad.py` &mdash; CLI entry point that loads `wesad/config.yaml`, trains the DNN, and optionally saves metrics/checkpoints.
- `data/WESAD/raw/` &mdash; raw WESAD dataset (symlink or copy the original release here).
- `data/processed/` &mdash; feature-extracted datasets (e.g., `wesad/hf_dataset`).

## Typical Workflow
1. **Preprocess WESAD:** `python wesad/preprocess.py run --path data/WESAD/raw --out_dir data/processed/wesad`
2. **Train / Evaluate:** `python scripts/train_wesad.py`  
   Override any setting on the fly, e.g. `python scripts/train_wesad.py --override training.batch_size=128`.

## TODO
- Document preprocessing/feature schemas for upcoming datasets.
- Add automated tests for data preparation and training loops.
