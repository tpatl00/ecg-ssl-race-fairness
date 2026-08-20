import os
import sys
import torch

_HPC_DIR = os.path.dirname(os.path.abspath(__file__))
if _HPC_DIR not in sys.path:
    sys.path.insert(0, _HPC_DIR)

from risk_prediction_base_model_hpc import RiskPredictionModel
from ecg_fm_randinit_hpc import ECGFM_RandInit_Encoder


class ECGFM_E2E_RiskPredictionModel(RiskPredictionModel):
    """
    End-to-end random-init ECG-FM for HF risk prediction.

    Architecture:
        Raw ECG (B, 12, 5000) -> ECGFM_RandInit_Encoder -> (B, 768)
                              -> downstream_net (Linear 768->256->1) -> (B, 1)

    All ~90.9M encoder parameters are trainable.
    """

    def __init__(
        self,
        model_config: dict,
        optimizer_config: dict,
        ecg_fm_checkpoint_path: str,
        ecg_fm_fairseq_dir: str,
    ):
        super().__init__(
            model_config=model_config,
            optimizer_config=optimizer_config,
        )

        self.encoder = ECGFM_RandInit_Encoder(
            checkpoint_path=ecg_fm_checkpoint_path,
            fairseq_dir=ecg_fm_fairseq_dir,
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model B (ECG-FM random-init e2e) — total parameters:     {total:,}")
        print(f"Model B (ECG-FM random-init e2e) — trainable parameters: {trainable:,}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        latent = self.encoder(x)                 # (B, 12, 5000) -> (B, 768)
        if self.training and torch.rand(1).item() < 0.02:           # ~2% of steps
            with torch.no_grad():
                l = latent.detach().float()
                print(f"[latent] mean={l.mean().item():+.4f} std={l.std().item():.4f} "
                      f"batch_std_of_per_sample_mean={l.mean(dim=1).std().item():.4f} "
                      f"frac_dead_dims={(l.abs().mean(dim=0) < 1e-4).float().mean().item():.3f}")
        log_risk = self.downstream_net(latent)   # (B, 768) -> (B, 1)
        return log_risk
