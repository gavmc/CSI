import torch
import torch.nn as nn


class CSI_DETR(nn.Module):
    def __init__(
            self, 
            n_reciv: int = 4, 
            win_len: int = 50, 
            n_pred: int = 13, 
            d_model: int = 256, 
            n_heads: int = 8, 
            enc_depth: list = [2, 2], 
            dec_depth: int = 3,
            n_freq: int = 8
        ):
        super().__init__()


        self.time_embedding = nn.Parameter(torch.randn(1, win_len, 1, 1, d_model) * 0.02)
        self.rx_embedding = nn.Parameter(torch.randn(1, 1, n_reciv, 1, d_model) * 0.02)
        self.freq_embedding = nn.Parameter(torch.randn(1, 1, 1, n_freq, d_model) * 0.02)

        self.freq_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(32, d_model, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )

        self.freq_norm = nn.LayerNorm(d_model)
        self.freq_downsample = nn.AdaptiveAvgPool1d(n_freq)

        rx_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.rx_encoder = nn.TransformerEncoder(rx_encoder_layer, enc_depth[0])
        

        temp_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.temp_encoder = nn.TransformerEncoder(temp_encoder_layer, enc_depth[1])


        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, dec_depth)

        self.learned_queries = nn.Parameter(
            torch.rand(1, n_pred, d_model) * 0.02,
            requires_grad = True
        )

        self.conf_proj = nn.Linear(d_model, 1)
        self.pos_proj = nn.Linear(d_model, 2)


    def encode(self, x):
        B, T, R, S = x.shape # [B, T, R, S]

        x = x.reshape(B*T*R, 1, S)   #  [B*T*R, 1, S]
        x = self.freq_encoder(x)   #    [B*T*R, D, 13]
        x = self.freq_downsample(x) #   [B*T*R, D, K]

        _, D, K = x.shape
        x = x.transpose(1, 2) #         [B*T*R, K, D]
        x = self.freq_norm(x) #         [B*T*R, K, D]
        x = x.reshape(B, T, R, K, D)  # [B, T, R, K, D]

        x += self.time_embedding[:, :T] + self.rx_embedding[:, :, :R] + self.freq_embedding[:, :, :, :K]
        x = x.permute(0, 1, 3, 2, 4)  # [B, T, K, R, D]

        x = x.reshape(B*T*K, R, D) #    [B*T*K, R, D]
        x = self.rx_encoder(x) #        [B*T*K, R, D]
        x = x.reshape(B, T, K, R, D)  # [B, T, K, R, D]

        x = x.permute(0, 2, 3, 1, 4) #  [B, K, R, T, D]
        x = x.reshape(B*R*K, T, D) #    [B*R*K, T, D]
        x = self.temp_encoder(x) #      [B*R*K, T, D]
        x = x.reshape(B, K, R, T, D)  # [B, K, R, T, D]

        x = x.permute(0, 3, 1, 2, 4) #   [B, T, K, R, D]
        x = x.reshape(B, T * K * R, D) # [B, T*K*R, D]

        return x


    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        B, _, _ = x.shape

        q = self.learned_queries.expand(B, -1, -1)
        x = self.decoder(q, x)

        conf_pred = self.conf_proj(x).sigmoid()
        pos_pred = self.pos_proj(x)

        return torch.cat([pos_pred, conf_pred], dim=-1)
    



class SimpleLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, conf_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.conf_weight = conf_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pos_pred = pred[..., :2]
        conf_pred = pred[..., 2:3]

        pos_gt = target[..., :2]
        conf_gt = target[..., 2:3]

        conf_loss = self.bce(conf_pred, conf_gt)

        visible = (conf_gt > 0.5).expand_as(pos_pred)
        raw_pos_loss = self.smooth_l1(pos_pred, pos_gt)

        visible_count = visible.sum().clamp_min(1)
        pos_loss = (raw_pos_loss * visible).sum() / visible_count

        return self.pos_weight * pos_loss + self.conf_weight * conf_loss


if __name__ == "__main__":

    model = CSI_DETR()
    loss = SimpleLoss()

    t = torch.rand(1, 50, 4, 52)

    gt = torch.rand(1, 13, 3)

    out = model(t)
    l = loss(out, gt)

    params = 0

    for p in model.parameters():
        params += p.numel()

    print("Model params", params)
    print()
    print(out)
    print()
    print(l)



