import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from lifelines.utils import concordance_index
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd



class RiskPredictionModel(pl.LightningModule):
    """
    Downstream risk prediction head.
    """
    def __init__(
            self,
            model_config: dict,
            optimizer_config: dict,
    ):

        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])

        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.outputs = None


        # Downstream network
        hidden_dim = model_config.get("hidden_dim", 256)
        dropout = model_config.get("dropout", 0.3)

        self.downstream_net = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),   # single log-risk score
        )

        # Storage for epoch-level metric accumulation
        self._val_outputs = []
        self._test_outputs = []




    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (Batch, 768) pre-computed ECG embedding.
            mask: Unused, kept for interface compatibility.
        Returns:
            log_risk: (Batch, 1) predicted log-risk scores.
        """

        latent = x
        log_risk = self.downstream_net(latent)  # (Batch, 1)
        return log_risk

    def _common_step(self, batch: dict, stage: str) -> dict:
        """
        Unpacks batch, runs forward pass, computes Cox PH loss.
        """
        x = batch["ecg_waveform"]  # (B, 12, 5000)
        events = batch["eventStatus"].float()  # (B,)
        durations = batch["followUpTime"].float()  # (B,)
        mask = batch.get("mask", None)

        log_risk = self(x, mask=mask).squeeze(-1)  # (B,)

        loss = self.cox_loss(log_risk, durations, events)
        self.log(f"{stage}/cox_loss", loss, on_step=True, on_epoch=True, prog_bar=True)


        return {
            "loss": loss,
            "log_risk": log_risk.detach().cpu(),
            "events": events.detach().cpu(),
            "durations": durations.detach().cpu(),
        }


    def predict_step(self, batch, batch_idx):
        x = batch['ecg_waveform']
        log_risk = self(x).squeeze(-1)
        return pd.DataFrame({
            'log_risk': log_risk.detach().cpu().numpy(),
            'event': batch['eventStatus'].cpu().numpy(),
            'duration': batch['followUpTime'].cpu().numpy(),
            'patient_id': batch['patient_id'].cpu().numpy(),
        })


    # Training
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        out = self._common_step(batch, stage="train")
        return out["loss"]

    # Validation
    def validation_step(self, batch: dict, batch_idx: int):
        out = self._common_step(batch, stage="val")
        self._val_outputs.append(out)

    def on_validation_epoch_end(self):
        self._on_epoch_end(self._val_outputs, stage="val")
        self._val_outputs.clear()

    # Test
    def test_step(self, batch: dict, batch_idx: int):
        out = self._common_step(batch, stage="test")
        self._test_outputs.append(out)

    def on_test_epoch_end(self):
        self._on_epoch_end(self._test_outputs, stage="test")
        self._test_outputs.clear()




    # Epoch-level C-Index
    def _on_epoch_end(self, outputs: list, stage: str):
        """Aggregate predictions across all batches and compute Lifelines C-Index."""
        if not outputs:
            return

        log_risks = torch.cat([o["log_risk"]  for o in outputs]).numpy()
        events = torch.cat([o["events"]    for o in outputs]).numpy()
        durations = torch.cat([o["durations"] for o in outputs]).numpy()

        # Lifelines concordance_index: higher predicted risk → shorter survival
        # so negate log_risk to align sign convention.
        try:
            ci = concordance_index(
                event_times = durations,
                predicted_scores= -log_risks, # negate: higher risk = shorter time
                event_observed = events,
            )
        except Exception as e:
            print(f"[{stage}] C-Index computation failed: {e}")
            ci = float("nan")

        self.outputs = pd.DataFrame({
            'log_risk': log_risks,
            'event': events,
            'duration': durations,
        })

        self.log(f"{stage}/c_index", ci, on_epoch=True, prog_bar=True)
        print(f"\n[{stage}] C-Index: {ci:.4f}")


    def configure_optimizers(self):
        lr = self.optimizer_config.get("lr", 1e-4)
        weight_decay = self.optimizer_config.get("weight_decay", 1e-5)
        warmup_steps = self.optimizer_config.get("warmup_steps", 100)
        max_steps = self.optimizer_config.get("max_steps", 1000)

        # Only optimise parameters that require gradients
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

        # Cosine annealing with linear warmup via LambdaLR -> from Dr Chen's work
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, max_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]















    # Implemented from the work of Xuelong An Wang
    def cox_loss(self, log_risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor):
        """
        Compute Cox proportional hazards loss.

        Args:
            log_risk (Tensor): Theta tensor, shape (batch_size, 1).
            events (Tensor): Event indicator tensor, shape (batch_size, 1).
            durations (Tensor): Time tensor, shape (batch_size, 1). Event time if uncensored,
                                    censoring time if censored.

        Returns:
            Tensor: Cox loss.
        """
        if not self.is_valid_cox_batch(events, durations):
            return log_risk.sum() * 0

        time = durations.reshape(-1)
        theta = log_risk.reshape(-1, 1)
        risk_mat = (time >= time[:, None]).float()
        loss_cox = (
                           theta.reshape(-1) - self.logsumexp(theta.T, mask=risk_mat, dim=1)
                   ) * events.reshape(-1)
        loss_cox = loss_cox.sum() / events.sum()
        return -loss_cox

    def is_valid_cox_batch(self, delta: torch.Tensor, time: torch.Tensor) -> bool:
            """
            Check if the batch is valid for computing Cox loss.
            In the cases below, loss is not defined
            1. If there is no uncensored data
            2. If there are uncensored samples but the risk matrix is empty.
              I.e., no censored patients survived more that the event time of any
                uncensored patient.

            Args:
                delta (Tensor): Event indicator tensor, shape (batch_size, 1).
                time (Tensor): Survival time tensor, shape (batch_size, 1).

            Returns:
                bool: True if valid, False otherwise.
            """
            risk_matrix = (time >= time[:, None]).float()
            return ((risk_matrix.sum(dim=1) * delta.reshape(-1)) > 1).any()


    def logsumexp(self, input_tensor: torch.Tensor, mask: torch.Tensor = None, dim: int = None, keepdim: bool = False):
        """
        Compute the log of the sum of exponentials of input elements (masked).

        Args:
            input_tensor (Tensor): Input tensor.
            mask (Tensor, optional): Mask tensor, same shape as x.
            dim (int, optional): Dimension to reduce.
            keepdim (bool, optional): Keep dimension.

        Returns:
            Tensor: Result tensor.
        """
        if dim is None:
            input_tensor, dim = input_tensor.view(-1), 0
        max_value, _ = torch.max(input_tensor, dim=dim, keepdim=True)
        input_tensor = input_tensor - max_value
        res = torch.exp(input_tensor)
        if mask is not None:
            res = res * mask
        res = torch.log(torch.sum(res, dim=dim, keepdim=keepdim) + 1e-8)
        return res + max_value.squeeze(dim) if not keepdim else res + max_value