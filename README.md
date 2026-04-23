# Explainable AI for Fall Detection using Advanced Machine Learning

Source code for KCL 6CCS3PRJ dissertation by Sophie Barberon, 2026.

## Overview

This project develops an explainable fall detection pipeline on the UP-Fall dataset. Four architectures are compared under Leave-One-Subject-Out cross-validation (SVM, 1D-CNN, BiLSTM-Attention, ST-GCN). The BiLSTM is analysed using SHAP feature attribution and temporal attention, and structured XAI outputs are converted to natural-language narratives via Anthropic's Claude Sonnet 4.5.

## Repository Structure

- `src/` — main training scripts and data preparation for SVM, CNN, BiLSTM (5.6-second windows)
- `stgcn/` — ST-GCN implementation (separate due to its graph-structured input shape, 2-second windows)
- `notebooks/` — Jupyter notebooks for SHAP, attention, LLM narratives, and per-subject failure diagnostics
- `outputs/` — result JSON files and generated figures referenced in the dissertation

## Dataset

The UP-Fall dataset is publicly available at https://sites.google.com/up.edu.mx/har-up/.
Download `CompleteDataSet.csv` and place it in `data/` before running `prepare_upfall.py`.

## Running the Pipeline

1. `python src/prepare_upfall.py` — windows the raw CSV into 5.6-second segments
2. `python src/add_activities.py` — reconstructs per-window activity labels
3. `python src/train_svm.py` — SVM LOSO with C grid search
4. `python src/train_cnn.py` — 1D-CNN residual LOSO
5. `python src/train_lstm.py` — BiLSTM-Attention LOSO
6. `python src/train_lstm_random.py` — BiLSTM random-split (for protocol comparison)
7. `python stgcn/monique.py` — ST-GCN LOSO (uses separate 2-second windowing)
8. Open notebooks in `notebooks/` for SHAP, attention, LLM narratives, and diagnostics

## Dependencies

See `requirements.txt`. Key libraries: PyTorch 2.x, scikit-learn, SHAP, anthropic, pandas, numpy, matplotlib.

## API Key

The LLM narrative notebook requires an Anthropic API key:

export ANTHROPIC_API_KEY="your-key-here"


## Author

Sophie Barberon, King's College London. Final Year Project 2026.
