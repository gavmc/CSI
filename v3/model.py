import torch
import torch.nn as nn


class CSI_DETR(nn.Module):
    def __init__(
            self, 
            n_waves: int = 52, 
            n_reciv: int = 4, 
            win_len: int = 50, 
            n_pred: int = 13, 
            d_model: int = 512, 
            n_heads: int = 8, 
            enc_depth: int = 3, 
            dec_depth: int = 3
        ):
        super().__init__()


        self.pos_embeddings = nn.Parameter(
            torch.rand(1, win_len, 1, n_waves) * 0.02,
            requires_grad=True
        )

        self.rx_embeddings = nn.Parameter(
            torch.rand(1, win_len, n_reciv, 1) * 0.02,
            requires_grad=True
        )

        self.data_encoder = nn.Sequential(
            nn.LayerNorm(n_waves * n_reciv),
            nn.Linear(n_waves * n_reciv, d_model)
        )
        

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_model*4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            enc_depth
        )


        decoder_layer = nn.TransformerDecoderLayer(
            d_model,
            n_heads,
            d_model*4,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            dec_depth,
        )

        self.learned_queries = nn.Parameter(
            torch.rand(1, n_pred, d_model) * 0.02,
            requires_grad = True
        )

        self.conf_proj = nn.Linear(d_model, 1)
        self.pos_proj = nn.Linear(d_model, 2)



    def forward(self, x: torch.Tensor):
        B, L, R, W = x.shape
        x += self.pos_embeddings + self.rx_embeddings
        x = x.reshape(B, L, R*W)

        x = self.data_encoder(x)

        x = self.encoder(x)
        x = self.decoder(self.learned_queries.expand(B, -1, -1), x)

        pos_pred = self.pos_proj(x)
        conf_pred = self.conf_proj(x)

        return torch.cat([pos_pred, conf_pred], dim=-1)
    



class Simple_Loss(nn.Module):
    def __init__(self, pos_weight, conf_weight):
        super().__init__()

        self.pos_weight = pos_weight
        self.conf_weight = conf_weight
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")
        self.BCE = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(self, x, y):
        pos_pred = x[:, :, :2]
        conf_pred = x[:, :, 2:3]

        pos_gt = y[:, :, :2]
        conf_gt = y[:, :, 2:3]

        conf_loss = self.BCE(conf_pred, conf_gt)
        pos_loss = self.smooth_l1(pos_pred, pos_gt)

        conf_mask = conf_gt > 0.5
        conf_mask = conf_mask.expand_as(pos_loss)

        pos_loss = (pos_loss * conf_mask).sum()

        return self.pos_weight*pos_loss + self.conf_weight*conf_loss


        
    






