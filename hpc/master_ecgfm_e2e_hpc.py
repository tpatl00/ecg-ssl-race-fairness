
import argparse
import os
import sys

import numpy as np
import torch
import torch.utils.data
import yaml
from sklearn.model_selection import StratifiedKFold

torch.set_float32_matmul_precision('high')

_HPC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HPC_DIR not in sys.path:
    sys.path.insert(0, _HPC_DIR)

from stratified_cv_hpc import stratified_train_and_evaluate
from my_HEEDB_waveform_dataset_hpc import HEEDBWaveformDataset
from ecgfm_e2e_risk_pred_model_hpc import ECGFM_E2E_RiskPredictionModel


def main(args):
    with open(args.config_dir, "r") as f:
        config = yaml.safe_load(f)

    if args.max_epochs is not None:
        config['trainer_config']['max_epochs'] = args.max_epochs
        print(f"[CLI override] max_epochs = {args.max_epochs}")

    if args.warmup_steps is not None:
        config['optimizer_config']['warmup_steps'] = args.warmup_steps
        print(f"[CLI override] warmup_steps = {args.warmup_steps}")

    if args.lr is not None:
        config['optimizer_config']['lr'] = args.lr
        print(f"[CLI override] lr = {args.lr}")

    if args.experiment_name:
        config['wandb']['run_name_prefix'] = args.experiment_name
        print(f"[CLI override] experiment_name = {args.experiment_name}")

    if args.enable_wandb:
        config['wandb']['disable_logging'] = False
        print("[CLI override] WandB logging enabled")

    fold_index = args.fold_index

    dataset = HEEDBWaveformDataset(config["data_dir"])
    labels = dataset.labels[:, 0]

    skf = StratifiedKFold(
        n_splits=config['n_splits'],
        shuffle=True,
        random_state=config['random_seed'],
    )
    all_splits = list(skf.split(np.zeros(len(dataset)), labels))

    if fold_index < 1 or fold_index > len(all_splits):
        print(f"ERROR: Fold index {fold_index} invalid for {len(all_splits)} splits.")
        sys.exit(1)

    train_indices, val_indices = all_splits[fold_index - 1]

    if args.smoke_n is not None:
        rng = np.random.default_rng(config['random_seed'])
        smoke_train_n = int(args.smoke_n * 0.9)
        smoke_val_n = args.smoke_n - smoke_train_n

        def _stratified_subsample(indices, n, rng):
            y = labels[indices]
            pos = indices[y == 1]
            neg = indices[y == 0]
            n_pos = int(round(n * len(pos) / len(indices)))
            n_pos = min(n_pos, len(pos))
            n_neg = min(n - n_pos, len(neg))
            pos_sel = rng.choice(pos, size=n_pos, replace=False)
            neg_sel = rng.choice(neg, size=n_neg, replace=False)
            out = np.concatenate([pos_sel, neg_sel])
            rng.shuffle(out)
            return out

        train_indices = _stratified_subsample(train_indices, smoke_train_n, rng)
        val_indices = _stratified_subsample(val_indices, smoke_val_n, rng)
        print(
            f"[smoke] subsampled to {len(train_indices)} train / {len(val_indices)} val; "
            f"event_rate train={labels[train_indices].mean():.3f} "
            f"val={labels[val_indices].mean():.3f}"
        )

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    max_iters = (
        config['trainer_config']['max_epochs']
        * len(train_subset)
        // config['dataloader_config']['batch_size']
    )
    config['optimizer_config']['max_steps'] = max_iters

    risk_model = ECGFM_E2E_RiskPredictionModel(
        model_config=config['model_config'],
        optimizer_config=config['optimizer_config'],
        ecg_fm_checkpoint_path=config['ecg_fm']['checkpoint_path'],
        ecg_fm_fairseq_dir=config['ecg_fm']['fairseq_path'],
    )

    results = stratified_train_and_evaluate(
        config=config,
        model=risk_model,
        train_dataset=train_subset,
        val_dataset=val_subset,
        fold_num=fold_index,
    )

    print(f"Fold {fold_index} finished. Val results: {results}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model B: random-init ECG-FM end-to-end")
    parser.add_argument(
        "--config_dir",
        default=os.path.join(_HPC_DIR, "ecgfm_e2e_config_hpc_hpc.yaml"),
    )
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Override optimizer_config.warmup_steps from the yaml (useful for smoke tests).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override optimizer_config.lr from the yaml (e.g. --lr 1e-5 for smoke tests).",
    )
    parser.add_argument(
        "--smoke_n",
        type=int,
        default=None,
        help="If set, subsample to this total N (stratified by event) for a fast smoke test. "
             "Split 90/10 train/val inside the chosen fold. Bypasses the full dataset.",
    )
    args = parser.parse_args()
    main(args)
