import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CSIPoseTransformer(nn.Module):
    def __init__(
        self,
        n_receivers: int = 4,
        n_subcarriers: int = 52,
        window_len: int = 50,
        d_model: int = 96,
        n_heads: int = 4,
        n_temporal_layers: int = 3,
        n_fusion_layers: int = 2,
        n_decoder_layers: int = 2,
        n_joints: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.R, self.T = n_receivers, window_len

        self.frame_embed = nn.Sequential(
            nn.LayerNorm(n_subcarriers),
            nn.Linear(n_subcarriers, d_model),
        )

        self.pos_embed = nn.Parameter(torch.randn(1, 1, window_len, d_model) * 0.02)
        self.rx_embed = nn.Parameter(torch.randn(1, n_receivers, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, n_temporal_layers)

        fus_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fus_layer, n_fusion_layers)

        self.joint_queries = nn.Parameter(torch.randn(1, n_joints, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, R, T, S = x.shape
        tok = self.frame_embed(x)
        tok = tok + self.pos_embed + self.rx_embed

        tok = tok.reshape(B * R, T, -1)
        tok = self.temporal_encoder(tok)
        tok = tok.reshape(B, R * T, -1)
        return self.fusion(tok)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = self.encode(x)
        q = self.joint_queries.expand(x.shape[0], -1, -1)
        out = self.decoder(q, mem)
        out = self.head(out)
        xy = torch.sigmoid(out[..., :2])
        conf = out[..., 2:]
        return torch.cat([xy, conf], dim=-1)


class MaskedCSIPretrainer(nn.Module):
    def __init__(self, backbone: CSIPoseTransformer, n_subcarriers: int = 52,
                 mask_ratio: float = 0.4):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        d = backbone.joint_queries.shape[-1]
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.recon_head = nn.Linear(d, n_subcarriers)

    def forward(self, x: torch.Tensor):
        B, R, T, S = x.shape
        tok = self.backbone.frame_embed(x) + self.backbone.pos_embed + self.backbone.rx_embed

        mask = torch.rand(B, R, T, device=x.device) < self.mask_ratio
        tok = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tok), tok)

        tok = tok.reshape(B * R, T, -1)
        tok = self.backbone.temporal_encoder(tok)
        tok = self.backbone.fusion(tok.reshape(B, R * T, -1))

        recon = self.recon_head(tok).reshape(B, R, T, S)
        loss = F.mse_loss(recon[mask], x[mask])
        return loss


def keypoint_loss(pred: torch.Tensor, target: torch.Tensor,
                  vis_mask: torch.Tensor) -> torch.Tensor:
    coord_loss = F.smooth_l1_loss(pred[..., :2][vis_mask], target[vis_mask])
    conf_loss = F.binary_cross_entropy_with_logits(pred[..., 2], vis_mask.float())
    return coord_loss + 0.2 * conf_loss


def resample_to_grid(timestamps_us: np.ndarray, amps: np.ndarray,
                     grid_us: np.ndarray) -> np.ndarray:
    out = np.empty((len(grid_us), amps.shape[1]), dtype=np.float32)
    for s in range(amps.shape[1]):
        out[:, s] = np.interp(grid_us, timestamps_us, amps[:, s])
    return out


if __name__ == "__main__":
    model = CSIPoseTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")
    x = torch.randn(8, 4, 50, 52)
    print("output:", model(x).shape)
