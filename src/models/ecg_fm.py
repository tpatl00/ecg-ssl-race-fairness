import torch
import torch.nn as nn
from omegaconf import OmegaConf


class ECGFM_Encoder(nn.Module):
    def __init__(self, checkpoint_path, fairseq_dir):
        super().__init__()

        # Dynamically add fairseq to path
        import sys
        if fairseq_dir not in sys.path:
            sys.path.append(fairseq_dir)

        from fairseq_signals import tasks, models

        # Load state
        state = torch.load(checkpoint_path, map_location="cpu")
        raw_cfg = state["cfg"]

        # Clean up legacy keys
        if isinstance(raw_cfg, dict) and "task" in raw_cfg:
            raw_cfg["task"].pop("model_name", None)
            raw_cfg["task"].pop("clocs_mode", None)

        # Wrap in OmegaConf
        cfg = OmegaConf.create(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
        if not isinstance(raw_cfg, dict):
            OmegaConf.set_struct(cfg, False)
            if hasattr(cfg, "task"):
                cfg.task.pop("model_name", None)
                cfg.task.pop("clocs_mode", None)

        # Build Task & Model
        task = tasks.setup_task(cfg.task)
        self.model = task.build_model(cfg.model)

        # Load weights
        self.model.load_state_dict(state["model"], strict=False)
        print("Successfully loaded ECG-FM Pretrained Weights natively.")

    def forward(self, x, mask=None, **kwargs):
        # x shape: (Batch, Channels, Time) -> e.g., (B, 12, 5000)

        # Split 10s into two 5s segments
        midpoint = x.shape[-1] // 2
        seg1 = x[:, :, :midpoint]
        seg2 = x[:, :, midpoint:]

        # Stack along batch dim: (B*2, 12, 2500)
        x = torch.cat([seg1, seg2], dim=0)

        # Extract features
        res = self.model.extract_features(source=x, padding_mask=None)
        features = res["x"]
        padding_mask = res["padding_mask"]

        # Mean Pooling
        if padding_mask is not None and padding_mask.any():
            features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            lengths = (~padding_mask).sum(dim=1, keepdim=True).float()
        else:
            lengths = torch.tensor(features.shape[1], dtype=torch.float, device=features.device)

        pooled_features = features.sum(dim=1) / lengths.clamp(min=1e-9)

        # Split back and Aggregate (Embedding-level mean)
        seg1_feats, seg2_feats = torch.chunk(pooled_features, 2, dim=0)
        pooled_features = (seg1_feats + seg2_feats) / 2.0

        return pooled_features