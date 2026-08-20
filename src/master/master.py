import argparse
import os
import yaml
import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys

from src.classes.heedb_dataloader import HEEDBDataset
from src.risk_models.risk_pred_base_model import RiskPredictionModel
from src.risk_models.resnet_risk_pred_model import ResNetRiskPredictionModel
from src.utils.stratified_cv import stratified_train_and_evaluate

def main(args):

    # Load the correct config depending on model
    CONFIG_PATHS = {
        "ecgfm_linear": "src/config/ecg_fm_config.yaml",
        "resnet_baseline": "src/config/resnet_baseline_config.yaml",
    }

    config_path = os.path.join(_REPO_ROOT, CONFIG_PATHS[args.experiment])

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply overrides from the CLI args
    if args.experiment_name:
        config['wandb']['run_name_prefix'] = args.experiment_name
        print(f"[CLI override] experiment_name = {args.experiment_name}")

    if args.enable_wandb:
        config['wandb']['disable_logging'] = False
        print("[CLI override] WandB logging enabled")

    if args.max_epochs is not None:
        config['trainer_config']['max_epochs'] = args.max_epochs
        print(f"[CLI override] max_epochs = {args.max_epochs}")

    # Set tf precision to high for the resnet training
    if args.experiment == "resnet_baseline":
        torch.set_float32_matmul_precision('high')


    # Build datasets depending on the experiment running
    if args.experiment == "ecgfm_linear":
        dataset = HEEDBDataset(config["data_dir"], mode="embedding",
                                      embedding_file=args.embedding_file)
    else:  # resnet_baseline
        dataset = HEEDBDataset(config["data_dir"], mode="waveform",
                                      waveform_dtype="float16")

    fold_index = args.fold_index


    labels = dataset.labels[:, 0]

    # Split dataset into train and val folds
    strat_k_fold = StratifiedKFold(
        n_splits=config['n_splits'],
        shuffle=True,
        random_state=config['random_seed'],
    )

    all_splits = list(strat_k_fold.split(np.zeros(len(dataset)), labels))

    # Check fold index in range
    if fold_index < 1 or fold_index > len(all_splits):
        print(f"ERROR: Fold index {fold_index} is invalid or out of bounds for {len(all_splits)} splits.")
        sys.exit(1)

    # Extract train and val subsets
    train_indices, val_indices = all_splits[fold_index - 1]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    # Compute the max steps for optimiser
    max_steps = (
        config['trainer_config']['max_epochs'] * len(train_subset) // config['dataloader_config']['batch_size']
    )

    config['optimizer_config']['max_steps'] = max_steps

    if args.experiment == "ecgfm_linear":
        risk_model = RiskPredictionModel(model_config= config['model_config'], optimizer_config=config['optimizer_config'])
    else:  # resnet_baseline
        risk_model = ResNetRiskPredictionModel(model_config=config['model_config'], optimizer_config=config['optimizer_config'])


    results = stratified_train_and_evaluate(
        config=config,
        model=risk_model,
        train_dataset=train_subset,
        val_dataset=val_subset,
        fold_num=fold_index,
    )

    print(f"Fold {fold_index} finished with Validation Results: {results}")





_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", required=True,
                        choices=["ecgfm_linear", "resnet_baseline"])
    parser.add_argument("--fold_index", type=int, required=True)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--embedding_file", type=str, default=None)


    args = parser.parse_args()
    main(args)



