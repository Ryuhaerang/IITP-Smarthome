# IITP-Smarthome

## Environment Setup
1. `conda create -n smarthome python=3.10`
2. `conda activate smarthome`
3. `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`
4. `python -m pip install numpy pandas neurokit2 fire tqdm scipy datasets scikit-learn`

## Repository Layout
- `wesad/` &mdash; preprocessing, data, model, trainer, and config utilities for the WESAD dataset.
- `scripts/train_wesad.py` &mdash; CLI entry point that loads `config/wesad/default.yaml`, trains a model, and saves metrics/checkpoints.
- `data/WESAD/raw/` &mdash; raw WESAD dataset.
- `data/processed/` &mdash; feature-extracted datasets.

## Usage
1. **Preprocess WESAD:** `python wesad/preprocess.py run --path data/WESAD/raw --out_dir data/processed/wesad`
2. **Train / Evaluate:** `python -m scripts.train_wesad`  
   Override settings on the fly, e.g. `python -m scripts.train_wesad --override training.batch_size=128`.
   Enable quantization by editing `config/wesad/default.yaml` (e.g., `quantization.enable_int8: true`).

## TODO
- Implement remaining two datasets (TBD).