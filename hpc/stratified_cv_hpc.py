import os
import yaml
import torch
import wandb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import sys
import random


def stratified_train_and_evaluate(config,
                                  model,
                                  train_dataset,
                                  val_dataset,
                                  fold_num):

    pl.seed_everything(config['random_seed'])
    g = torch.Generator().manual_seed(config['random_seed'])

    # --- 1. SETUP UNIQUE PATHS AND RESUME LOGIC ---
    run_name = f"{config['wandb']['run_name_prefix']}_fold_{fold_num}"
    project_name = config['wandb']['project_name']
    output_dir = Path("training_outputs") / project_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    eval_results_dir = output_dir / "eval_results"
    eval_results_dir.mkdir(parents=True, exist_ok=True)
    is_sweep = os.environ.get('WANDB_SWEEP_ID') is not None

    # W&B Resume Logic: Check if a run ID file exists
    run_id_file = output_dir / "wandb_run_id.txt"
    wandb_id = None
    if run_id_file.is_file():
        with open(run_id_file, "r") as f:
            wandb_id = f.read().strip()
        print(f"Resuming W&B run with ID: {wandb_id}")

    if not config['eval_only'] and not is_sweep:
        print("Resume from last checkpoint")
        # Checkpoint Resume Logic
        last_ckpt_path = checkpoint_dir / "last.ckpt"
        resume_from_checkpoint = str(last_ckpt_path) if last_ckpt_path.is_file() else None
        if resume_from_checkpoint:
            print(f"Resuming trainer state from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Starting new training run for fold {fold_num}.")

    else:
        print('Training from scratch OR hparam sweep')
        resume_from_checkpoint = None

    # Dataloaders
    train_batch_sampler = StratifiedBatchSampler(
        train_dataset.dataset.labels[train_dataset.indices, 0],
        batch_size=config['dataloader_config']['batch_size'],
        shuffle=True,
        random_state=config['random_seed']
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        num_workers=config['dataloader_config']['num_workers'],
        worker_init_fn=seed_worker,
        generator=g
    )

    temp_train_dataloader = DataLoader(
        train_dataset,
        # batch_size=config['data']['batch_size'],
        # num_workers=config['data']['num_workers'],
        **config['dataloader_config'],
        worker_init_fn=seed_worker,
        shuffle=False,
        generator=g
    )

    val_dataloader = DataLoader(
        val_dataset,
        # batch_size=config['data']['batch_size'],
        # num_workers=config['data']['num_workers'],
        **config['dataloader_config'],
        worker_init_fn=seed_worker,
        shuffle=False,
        generator=g
    )

    # pdb.set_trace()
    try:
        max_iters = config['trainer_config']['max_epochs'] * len(train_dataloader)
        # Inject the calculated value into the scheduler's parameters
        if 'scheduler' in config['optimizer_config']:
            config['optimizer_config']['scheduler']['params']['num_training_steps'] = max_iters
    except NameError:
        raise NameError("Make sure `train_dataloader` is defined before this step.")

    # --- Logger Setup with Sweep Support ---
    if wandb.run is not None:
        print(f"--> SWEEP DETECTED: Attaching to existing run {wandb.run.id}")
        wandb_logger = pl.loggers.WandbLogger(
            experiment=wandb.run,  # <--- CRITICAL: Use the run started by the Agent
            log_model=False
        )
    else:
        # Standard execution (No sweep)
        print(f"--> STANDARD CV: Initializing Run for Fold {fold_num}")
        wandb_logger = pl.loggers.WandbLogger(
            project=config['wandb']['project_name'],
            entity=config['wandb']['entity'],
            name=f"{config['wandb']['run_name_prefix']}_fold_{fold_num}",
            id=wandb_id,
            log_model=False,
            resume='allow'
        )
        # Only save ID if we created the run ourselves
        if wandb_logger.experiment.id and not run_id_file.is_file():
            with open(run_id_file, "w") as f:
                f.write(wandb_logger.experiment.id)



    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_c_index}',
        save_top_k=1,
        monitor="val/c_index",
        mode='max',
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    final_callbacks = [checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        **config['trainer_config'],
        logger=wandb_logger if not config['wandb']['disable_logging'] else None,
        callbacks=final_callbacks,
        precision="16-mixed"

        # resume_from_checkpoint=resume_from_checkpoint,
    )

    if not config['eval_only']:
        trainer.fit(model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=val_dataloader,
                    ckpt_path=resume_from_checkpoint,
                    weights_only=False
                    )

        # Test with the best checkpoint from validation
        val_results = trainer.validate(dataloaders=val_dataloader,
                                       ckpt_path="best",
                                       weights_only=False)
        # if model.outputs is not None:
        #     save_path = os.path.join(eval_results_dir, f"eval_preds_fold_{fold_num}.csv")
        #     model.outputs.to_csv(save_path, index=False)
        #
        # # Also generate train set predictions for the best model
        # train_results = trainer.predict(model=model,
        #                                 dataloaders=temp_train_dataloader,
        #                                 ckpt_path="best",
        #                                 weights_only=False)
        # train_save_path = os.path.join(eval_results_dir, f"train_preds_fold_{fold_num}.csv")
        # all_train_results = pd.concat([result for result in train_results])
        # # pdb.set_trace()
        # all_train_results.to_csv(train_save_path, index=False)

        # Save validation predictions
        # Val predictions
        val_preds = trainer.predict(model=model,
                                    dataloaders=val_dataloader,
                                    ckpt_path="best",
                                    weights_only=False)
        val_preds_df = pd.concat([result for result in val_preds])
        val_preds_df.to_csv(os.path.join(eval_results_dir, f"eval_preds_fold_{fold_num}.csv"), index=False)

        # Train predictions
        train_results = trainer.predict(model=model,
                                        dataloaders=temp_train_dataloader,
                                        ckpt_path="best",
                                        weights_only=False)
        all_train_results = pd.concat([result for result in train_results])
        all_train_results.to_csv(os.path.join(eval_results_dir, f"train_preds_fold_{fold_num}.csv"), index=False)
    else:
        # model.load_state_dict(torch.load(config['ensemble']['eval_checkpoint_paths'][fold_num]))
        model.eval()
        val_results = trainer.test(model=model,
                                   dataloaders=val_dataloader,
                                   ckpt_path=resume_from_checkpoint,
                                   weights_only=False)

        # if model.outputs is not None:
        #     save_path = os.path.join(eval_results_dir, f"last_eval_preds_fold_{fold_num}.csv")
        #     model.outputs.to_csv(save_path, index=False)
        val_preds = trainer.predict(model=model,
                                    dataloaders=val_dataloader,
                                    ckpt_path=resume_from_checkpoint,
                                    weights_only=False)
        val_preds_df = pd.concat([result for result in val_preds])
        val_preds_df.to_csv(os.path.join(eval_results_dir, f"last_eval_preds_fold_{fold_num}.csv"), index=False)

    if not is_sweep:
        # Standard CV: We created the run, so we must close it to upload artifacts.
        wandb.finish()
    else:
        # Sweep Mode: Do NOT finish here.
        # The run must remain open so the Agent can log final metrics or start next trial.
        pass

    return val_results



class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides same representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True,random_state=None):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        assert n_batches >= 1, 'number of batches must be at least 2'
        if n_batches>=2:
            self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        else: self.skf = None
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.random_state = random_state

    def __iter__(self):
        if self.shuffle and self.skf is not None:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        if self.skf is not None:
            for train_idx, test_idx in self.skf.split(self.X, self.y):
                yield test_idx
        else:
            a_list = torch.arange(len(self.y)).tolist()
            ## shuffle the list
            if self.shuffle:
                random.shuffle(a_list)
            yield a_list

    def __len__(self):
        return len(self.y) // self.batch_size

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)