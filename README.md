# Are ECG Foundation Models Always Better? A Study of Heart Failure Risk Prediction and Fairness

Tanay Patel, Xuelong An, and Chen Chen

*School of Computer Science, University of Sheffield, Sheffield, UK*

Accepted at the **AI in Healthcare (AIiH) 2026** conference as a long-form abstract.

[Long-form abstract on Zenodo](https://zenodo.org/records/21922076)

---

## Abstract

ECG foundation models which leverage self-supervised learning (SSL), can learn data-efficient, transferable representations from large-scale unlabelled ECG recordings to enable strong performance across a wide range of cardiac disease diagnostic tasks. However, few studies have explored the benefits of using foundation models for risk prediction tasks or whether they can improve demographic equity in these more challenging, yet clinically important, settings. To our knowledge, we present the first race-stratified fairness evaluation of a self-supervised ECG foundation model for incident heart failure (HF) risk prediction. We compare a supervised ResNet1D with an ECG foundation model (ECG-FM) using the Harvard-Emory ECG Database, with predictive performance measured by the concordance index (C-index) and fairness assessed across sex and racial subgroups using the Performance Gap (PG) and Coefficient of Variation (CV). Surprisingly, ResNet1D achieved the highest predictive performance, with a mean C-index of 0.800, significantly outperforming the pretrained ECG-FM (0.791) and the randomly initialised ECG-FM (0.789) (p < 0.05). In contrast, the pretrained ECG-FM yielded the smallest racial disparities, while the randomly initialised ECG-FM exhibited the largest, indicating that pretraining could reduce performance disparities across demographic subgroups. Sex-related disparities were very mild across all models, likely because the dataset had a balanced sex distribution. Our study demonstrates that demographic fairness in clinical risk prediction cannot be characterised by aggregate predictive performance alone. As racial imbalance remains common in clinical datasets, algorithmic innovations should be accompanied by routine demographic subgroup evaluation to ensure the fairness, reliability, and trustworthiness of AI-ECG models before clinical deployment.

## Repository Structure

```
ecg-ssl-race-fairness/
├── src/
│   ├── config/                     # Training configurations
│   │   ├── ecg_fm_config.yaml      # ECG-FM linear probe (Pretrained ECG-FM)
│   │   └── resnet_baseline_config.yaml  # ResNet1D baseline
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
│   │   └── master.py               # Training orchestrator
│   ├── utils/
│   │   ├── stratified_cv.py        # Stratified K-fold cross-validation
│   │   └── create_ecg_embeddings.py  # ECG-FM embedding extraction
│   └── analysis/
│       ├── compile_predictions.py  # Aggregate fold predictions + demographics
│       └── metrics_scripts/
│           ├── cindex.py           # C-index + pairwise Wilcoxon tests
│           └── fairness_metrics.py # PG, CV fairness metrics
├── hpc/                            # HPC training scripts (Random-Init ECG-FM)
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
│       └── fairness_metrics.csv    # PG, CV per model
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

### ResNet1D Supervised Baseline

```bash
python src/master/master.py --experiment resnet_baseline
```

### Pretrained ECG-FM (Frozen Linear Probe)

```bash
python src/master/master.py --experiment ecgfm_linear
```

### Random-Init ECG-FM End-to-End (HPC)

The Random-Init ECG-FM (90.9M parameters) requires GPU resources (trained on A100 80GB). On an HPC cluster with SLURM:

```bash
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

### Overall C-index (Stratified 10-Fold CV)

| Model | C-index | Std |
|-------|---------|-----|
| ResNet1D | 0.8004 | 0.0058 |
| Pretrained ECG-FM | 0.7905 | 0.0054 |
| Random-Init ECG-FM | 0.7885 | 0.0059 |

### Fairness Metrics (Race: White, Black, Asian)

| Model | Performance Gap (PG) | Coefficient of Variation (CV) |
|-------|---------------------|-------------------------------|
| ResNet1D | 0.0499 | 0.0253 |
| Pretrained ECG-FM | 0.0283 | 0.0146 |
| Random-Init ECG-FM | 0.0551 | 0.0298 |

ResNet1D achieved the highest overall C-index (0.8004), significantly outperforming both ECG-FM variants (p < 0.05, Holm-Bonferroni corrected). However, the Pretrained ECG-FM exhibited the smallest racial disparities (PG = 0.0283, CV = 0.0146), while the Random-Init ECG-FM had the largest, indicating that SSL pretraining contributes to more equitable performance across racial subgroups.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{patel2026ecgfm,
  title={Are {ECG} Foundation Models Always Better? {A} Study of Heart Failure Risk Prediction and Fairness},
  author={Patel, Tanay and An, Xuelong and Chen, Chen},
  booktitle={AI in Healthcare (AIiH)},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
