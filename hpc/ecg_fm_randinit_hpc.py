"""
Random-init ECG-FM encoder
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf


class ECGFM_RandInit_Encoder(nn.Module):
    def __init__(self, checkpoint_path: str, fairseq_dir: str):
        super().__init__()

        import sys
        if fairseq_dir not in sys.path:
            sys.path.append(fairseq_dir)

        from fairseq_signals import tasks

        state = torch.load(checkpoint_path, map_location="cpu")
        raw_cfg = state["cfg"]

        if isinstance(raw_cfg, dict) and "task" in raw_cfg:
            raw_cfg["task"].pop("model_name", None)
            raw_cfg["task"].pop("clocs_mode", None)

        cfg = OmegaConf.create(raw_cfg) if isinstance(raw_cfg, dict) else raw_cfg
        if not isinstance(raw_cfg, dict):
            OmegaConf.set_struct(cfg, False)
            if hasattr(cfg, "task"):
                cfg.task.pop("model_name", None)
                cfg.task.pop("clocs_mode", None)

        task = tasks.setup_task(cfg.task)
        self.model = task.build_model(cfg.model)

        # Deliberately NO load_state_dict — keeps PyTorch default init.
        del state
        print("ECGFM_RandInit_Encoder: architecture built, weights left at random init.")

    def forward(self, x, mask=None, **kwargs):
        print(f"[enc] mask_prob={getattr(self.model, 'mask_prob', None)} "
              f"training={self.model.training} "
              f"has_mask_emb={hasattr(self.model, 'mask_emb')}")
        midpoint = x.shape[-1] // 2
        seg1 = x[:, :, :midpoint]
        seg2 = x[:, :, midpoint:]

        x = torch.cat([seg1, seg2], dim=0)  # (2B, 12, 2500)

        res = self.model.extract_features(source=x, padding_mask=None, mask=False)
        features = res["x"]

        with torch.no_grad():
            per_sample = features.float().mean(dim=1)          # (B, 768)
            B = per_sample.shape[0]
            per_dim_batchstd = per_sample.std(dim=0).mean().item() if B > 1 else float('nan')
            if B >= 3:
                n = per_sample / per_sample.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                cos01 = (n[0] * n[1]).sum().item()
                cos02 = (n[0] * n[2]).sum().item()
                cos12 = (n[1] * n[2]).sum().item()
            else:
                cos01 = cos02 = cos12 = float('nan')
            print(f"[collapse] train={self.model.training} shape={tuple(features.shape)} "
                  f"per_dim_batchstd={per_dim_batchstd:.4f} "
                  f"cos(s0,s1)={cos01:.4f} cos(s0,s2)={cos02:.4f} cos(s1,s2)={cos12:.4f}")

        padding_mask = res["padding_mask"]

        if padding_mask is not None and padding_mask.any():
            features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            lengths = (~padding_mask).sum(dim=1, keepdim=True).float()
        else:
            lengths = torch.tensor(features.shape[1], dtype=torch.float, device=features.device)

        pooled = features.sum(dim=1) / lengths.clamp(min=1e-9)

        seg1_feats, seg2_feats = torch.chunk(pooled, 2, dim=0)
        return (seg1_feats + seg2_feats) / 2.0
