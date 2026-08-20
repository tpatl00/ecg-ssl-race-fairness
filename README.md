# ECG Self-Supervised Learning & Race Fairness

**Evaluating Self-Supervised ECG Representation Learning for Cardiac Risk Prediction: Performance and Fairness Across Racial Groups**

Tanay Patel, Venet Osmani

*University of Sheffield*

Published at the **AI in Healthcare Conference (AIHC 2025)** as a long-form abstract.

---

## Abstract

Self-supervised learning (SSL) offers a promising approach to learning generalisable ECG representations without relying on labelled data. Foundation models such as ECG-FM, pretrained on large-scale datasets using SSL objectives (SimCLR, BYOL, VICReg), have shown strong performance on downstream cardiac tasks. However, it remains unclear whether these learned representations encode or mitigate demographic biases present in clinical data.

This study evaluates SSL pretraining strategies for cardiac risk prediction and measures demographic fairness across racial groups using the Harvard-Emory ECG Database (HEEDB). We compare three modelling approaches:

| Model | Description |
|-------|-------------|
| **Model A** | ResNet1D supervised baseline (trained from scratch) |
| **Model B** | ECG-FM frozen linear probe (pretrained weights, frozen encoder) |
| **Model C** | ECG-FM random-init end-to-end (same architecture, no pretrained weights) |

All models are evaluated using stratified 10-fold cross-validation with Harrell's concordance index (C-index), Wilcoxon signed-rank tests with Holm-Bonferroni correction, and fairness metrics (Performance Gap, Coefficient of Variation, Worst-Group Performance).

## Repository Structure

```
ecg-ssl-race-fairness/
├── src/
│   ├── config/                     # Training configurations
│   │   ├── ecg_fm_config.yaml      # ECG-FM linear probe (Model B)
│   │   └── resnet_baseline_config.yaml  # ResNet baseline (Model A)
│   ├── models/
│   │   ├── ecg_fm.py               # ECG-FM encoder wrapper
│   │   └── resnet1d.py             # 1D ResNet encoder
│   ├── risk_models/
│   │   ├── risk_pred_base_model.py # Base PyTorch Lightning risk model
│   │   └── resnet_risk_pred_model.py  # ResNet risk prediction model
│   ├── preprocessing/
│   │   ├── ecg_preprocess.py       # ECG signal preprocessing
│   │   └── generate_preprocessed_ecg_memmap.py  # Memory-mapped dataset generation
│   ├── classes/
│   │   └── heedb_dataloader.py     # HEEDB PyTorch dataset & dataloader
│   ├── master/
│   │   └── master.py               # Training orchestrator (Models A & B)
│   ├── utils/
│   │   ├── stratified_cv.py        # Stratified K-fold cross-validation
│   │   └── create_ecg_embeddings.py  # ECG-FM embedding extraction
│   └── analysis/
│       ├── compile_predictions.py  # Aggregate fold predictions + demographics
│       └── metrics_scripts/
│           ├── cindex.py           # C-index + pairwise Wilcoxon tests
│           └── fairness_metrics.py # PG, CV, WGP fairness metrics
├── hpc/                            # HPC training scripts (Model C)
│   ├── master_ecgfm_e2e_hpc.py    # E2E training orchestrator
│   ├── ecg_fm_randinit_hpc.py     # Random-init ECG-FM encoder
│   ├── ecgfm_e2e_risk_pred_model_hpc.py  # E2E risk prediction model
│   ├── risk_prediction_base_model_hpc.py  # Base model (HPC variant)
│   ├── stratified_cv_hpc.py       # Stratified CV (HPC variant)
│   ├── my_HEEDB_waveform_dataset_hpc.py  # Waveform dataset (HPC variant)
│   ├── config/
│   │   └── ecgfm_e2e_config.yaml  # E2E HPC config (A100 GPU)
│   └── slurm/
│       └── submit_ecgfm_e2e_array.sh  # SLURM array job (10-fold)
├── results/
│   └── metrics/
│       ├── cindex_results.csv      # Per-model per-subgroup C-index
│       ├── cindex_pairwise.csv     # Pairwise Wilcoxon test results
│       └── fairness_metrics.csv    # PG, CV, WGP per model
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

This study uses the **Harvard-Emory ECG Database (HEEDB)**, which requires credentialed access through PhysioNet:

1. Create a PhysioNet account at [https://physionet.org](https://physionet.org)
2. Complete the required CITI training for credentialed access
3. Request access to the [Harvard-Emory ECG Database](https://physionet.org/content/ecg-mimic/)
4. Once approved, download the dataset and place it in `./data/harvard-emory-dataset/`

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tpatl00/ecg-ssl-race-fairness.git
cd ecg-ssl-race-fairness
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install fairseq-signals (required for ECG-FM models)

```bash
git clone https://github.com/Jwoo5/fairseq-signals.git
cd fairseq-signals
pip install -e .
cd ..
```

### 5. Download ECG-FM checkpoint

Download the MIMIC-IV ECG pretrained checkpoint from the [fairseq-signals repository](https://github.com/Jwoo5/fairseq-signals) and place it in `./checkpoints/`.

## Usage

### Data Preprocessing

Generate memory-mapped ECG arrays from the raw HEEDB dataset:

```bash
python src/preprocessing/generate_preprocessed_ecg_memmap.py
```

### Model A: ResNet1D Supervised Baseline

```bash
python src/master/master.py --experiment resnet_baseline
```

### Model B: ECG-FM Frozen Linear Probe

```bash
python src/master/master.py --experiment ecgfm_linear
```

### Model C: ECG-FM Random-Init End-to-End (HPC)

Model C requires GPU resources (trained on A100 80GB). On an HPC cluster with SLURM:

```bash
# Edit hpc/slurm/submit_ecgfm_e2e_array.sh with your username and email
sbatch hpc/slurm/submit_ecgfm_e2e_array.sh
```

### Evaluation

After training, compile predictions and compute metrics:

```bash
python src/analysis/compile_predictions.py
python src/analysis/metrics_scripts/cindex.py
python src/analysis/metrics_scripts/fairness_metrics.py
```

## Key Results

### Overall C-index (10-fold CV)

| Model | C-index | Std |
|-------|---------|-----|
| Model A (ResNet1D) | 0.800 | 0.006 |
| Model B (ECG-FM linear) | 0.790 | 0.005 |
| Model C (ECG-FM E2E) | 0.789 | 0.006 |

### Fairness Metrics (Race axis: White, Black, Asian)

| Model | Performance Gap | Coefficient of Variation | Worst-Group Performance |
|-------|----------------|--------------------------|------------------------|
| Model A (ResNet1D) | 0.050 | 0.025 | 0.787 |
| Model B (ECG-FM linear) | 0.028 | 0.015 | 0.782 |
| Model C (ECG-FM E2E) | 0.055 | 0.030 | 0.774 |

Model A (supervised ResNet) achieves the highest overall C-index. Model B (ECG-FM frozen linear probe) shows the smallest racial performance gap and lowest coefficient of variation, suggesting that SSL-pretrained representations may encode more equitable feature representations across racial groups.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{patel2025ecgssl,
  title={Evaluating Self-Supervised ECG Representation Learning for Cardiac Risk Prediction: Performance and Fairness Across Racial Groups},
  author={Patel, Tanay and Osmani, Venet},
  booktitle={AI in Healthcare Conference (AIHC)},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
