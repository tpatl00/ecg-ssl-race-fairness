import torch

from src.risk_models.risk_pred_base_model import RiskPredictionModel
from src.models.resnet1d import ResNet1D


class ResNetRiskPredictionModel(RiskPredictionModel):
    """
    End-to-end ResNet baseline for HF risk prediction.

    Raw ECG (B, 12, 5000) -> ResNet1D -> (B, 768) -> Cox head -> (B, 1)

    """

    def __init__(self, model_config: dict, optimizer_config: dict):
        # Initialise parent: sets up downstream_net (768->256->1), Cox loss, etc.
        super().__init__(
            model_config=model_config,
            optimizer_config=optimizer_config,
        )

        self.encoder = ResNet1D(
            in_channels=12, base_filters=64, kernel_size=16, stride=2,
            groups=1, n_block=8, n_classes=768,
            downsample_gap=2, increasefilter_gap=4,
        )

        # Report parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"CNN Baseline — total parameters: {total_params:,}")
        print(f"CNN Baseline — trainable parameters: {trainable_params:,}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (Batch, 12, 5000) raw 12-lead ECG waveform.
            mask: Unused, kept for interface compatibility.
        Returns:
            log_risk: (Batch, 1) predicted log-risk scores.
        """
        latent = self.encoder(x)               # (B, 12, 5000) -> (B, 768)
        log_risk = self.downstream_net(latent)  # (B, 768) -> (B, 1)
        return log_risk
