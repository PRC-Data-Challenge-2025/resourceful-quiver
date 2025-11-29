import torch
import torch.nn as nn
from torch.nn import functional as F

class TabularTransformer(nn.Module):
    def __init__(
        self,
        seq_input_dim,          # D_in: numeric + phase one-hot
        end_input_dim,          # D_end: len(TO_FEED_IN_THE_END)
        num_typecodes=26,       # from your fixed list
        type_emb_dim=16,        # CAT_EMBEDDING_DIM
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=3600,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Project per-timestep features -> d_model
        self.input_proj = nn.Linear(seq_input_dim, d_model)

        # Learnable [CLS] token (1 x 1 x d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional embeddings for [CLS] + up to max_seq_len steps
        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,   # we’ll feed [L, B, d_model]
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Typecode embedding (sequence-level)
        self.type_emb = nn.Embedding(num_typecodes, type_emb_dim)

        # Head: [CLS_out || type_emb || end_features] -> scalar
        head_in_dim = d_model + type_emb_dim + end_input_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, batch):
        """
        batch:
          - seq_features:   [B, L, D_in]
          - end_features:   [B, D_end]
          - typecode_id:    [B]
          - attention_mask: [B, L] (True = valid, False = pad)
        """
        x = batch["seq_features"]        # [B, L, D_in]
        end_feats = batch["end_features"]  # [B, D_end]
        typecode_id = batch["typecode_id"]  # [B]
        attn_mask = batch.get("attention_mask", None)  # [B, L] or None

        B, L, _ = x.shape

        # Project timestep features
        x = self.input_proj(x)           # [B, L, d_model]

        # Build CLS token for this batch
        cls = self.cls_token.expand(B, 1, self.d_model)  # [B, 1, d_model]

        # Concatenate CLS + sequence
        x = torch.cat([cls, x], dim=1)   # [B, L+1, d_model]

        # Positional encoding (same positions for all batch elems)
        # positions: [0 .. L] where 0 is CLS
        pos_ids = torch.arange(L + 1, device=x.device).unsqueeze(0).expand(B, L + 1)
        pos_enc = self.pos_emb(pos_ids)  # [B, L+1, d_model]
        x = x + pos_enc

        # Prepare attention mask for PyTorch Transformer:
        # src_key_padding_mask: [B, L+1], True where PAD.
        if attn_mask is not None:
            # Original mask: True=valid for [L]
            cls_valid = torch.ones(B, 1, dtype=attn_mask.dtype, device=attn_mask.device)
            full_valid = torch.cat([cls_valid, attn_mask], dim=1)      # [B, L+1]
            key_padding_mask = ~full_valid.bool()                      # True where pad
        else:
            key_padding_mask = None

        # Transformer expects [L+1, B, d_model]
        x = x.transpose(0, 1)            # [L+1, B, d_model]

        # Encode
        enc_out = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask,
        )                                # [L+1, B, d_model]

        # Take CLS output (position 0)
        cls_out = enc_out[0]             # [B, d_model]

        # Typecode embedding
        tc_emb = self.type_emb(typecode_id)  # [B, type_emb_dim]

        # Concatenate CLS + typecode_emb + end_features
        head_in = torch.cat([cls_out, tc_emb, end_feats], dim=-1)  # [B, head_in_dim]

        # Predict scalar (e.g. fuel_kg)
        out = self.head(head_in).squeeze(-1)  # [B]

        return out
