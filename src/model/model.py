import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

# Make DSTformer importable from src/model/pretrained/lib/
_pretrained_dir = Path(__file__).parent / "pretrained"
if str(_pretrained_dir) not in sys.path:
    sys.path.insert(0, str(_pretrained_dir))

from lib.model.DSTformer import DSTformer

# ---------------------------------------------------------------------------
# OpenPose body keypoint index → H36M keypoint index mapping
#
# OpenPose 25-keypoint body layout (every 3 values: x, y, conf):
#   0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist,
#   5=LShoulder, 6=LElbow, 7=LWrist, 8=MidHip,
#   9=RHip, 10=RKnee, 11=RAnkle, 12=LHip, 13=LKnee, 14=LAnkle,
#   15=REye, 16=LEye, 17=REar, 18=LEar,
#   19=LBigToe, 20=LSmallToe, 21=LHeel, 22=RBigToe, 23=RSmallToe, 24=RHeel
#
# H36M 17-keypoint layout (what MotionBERT expects):
#   0=Hip(root), 1=RHip, 2=RKnee, 3=RAnkle,
#   4=LHip, 5=LKnee, 6=LAnkle,
#   7=Spine, 8=Thorax, 9=Neck, 10=Head,
#   11=LShoulder, 12=LElbow, 13=LWrist,
#   14=RShoulder, 15=RElbow, 16=RWrist
#
# Mapping: H36M_idx → OpenPose_idx (-1 = synthesize from neighbours)
# ---------------------------------------------------------------------------
OPENPOSE_TO_H36M = {
    0:  8,   # Hip root  → MidHip
    1:  9,   # RHip      → RHip
    2:  10,  # RKnee     → RKnee
    3:  11,  # RAnkle    → RAnkle
    4:  12,  # LHip      → LHip
    5:  13,  # LKnee     → LKnee
    6:  14,  # LAnkle    → LAnkle
    7:  1,   # Spine     → Neck (approx)
    8:  1,   # Thorax    → Neck (approx)
    9:  1,   # Neck      → Neck
    10: 0,   # Head      → Nose
    11: 5,   # LShoulder → LShoulder
    12: 6,   # LElbow    → LElbow
    13: 7,   # LWrist    → LWrist
    14: 2,   # RShoulder → RShoulder
    15: 3,   # RElbow    → RElbow
    16: 4,   # RWrist    → RWrist
}


def extract_h36m_from_openpose(sequence: torch.Tensor) -> torch.Tensor:
    """
    Converts normalized OpenPose keypoints (B, T, 274) to H36M format (B, T, 17, 3).

    Input layout after drop_confidence (274 features):
        body:  25 joints × 2 = 50   → indices 0:50
        face:  70 joints × 2 = 140  → indices 50:190   (ignored)
        left:  21 joints × 2 = 42   → indices 190:232  (ignored)
        right: 21 joints × 2 = 42   → indices 232:274  (ignored)

    We only use the 25 body joints (x,y pairs) and remap to 17 H36M joints.
    The third channel (z) is set to 0 since we have 2D data.
    """
    B, T, _ = sequence.shape

    # Extract body xy: (B, T, 25, 2)
    body_xy = sequence[:, :, :50].reshape(B, T, 25, 2)

    # Build H36M tensor: (B, T, 17, 3) — z=0
    h36m = torch.zeros(B, T, 17, 3, device=sequence.device, dtype=sequence.dtype)
    for h36m_idx, op_idx in OPENPOSE_TO_H36M.items():
        h36m[:, :, h36m_idx, :2] = body_xy[:, :, op_idx, :]
        # z stays 0

    return h36m  # (B, T, 17, 3)


# ---------------------------------------------------------------------------
# PositionalEncoding (kept for the decoder)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 150, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# MotionBERT Encoder wrapper
# ---------------------------------------------------------------------------

class MotionBERTEncoder(nn.Module):
    """
    Wraps DSTformer (MotionBERT-Lite) as an encoder for sign language sequences.
    """

    CHECKPOINT = Path(__file__).parent / "pretrained" / "checkpoint" / "pretrain" / "MB_lite" / "latest_epoch.bin"

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1, freeze_epochs: int = 10):
        super().__init__()

        self.freeze_epochs = freeze_epochs
        self._current_epoch = 0

        self.dstformer = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=256,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            num_joints=17,
            maxlen=243,
            att_fuse=True,
        )

        # Fix checkpoint layer names mapping (removes 'module.' and 'model_pos.')
        if self.CHECKPOINT.exists():
            state = torch.load(self.CHECKPOINT, map_location="cpu")
            
            # Extract internal state dict
            if "model_pos" in state:
                ckpt_dict = state["model_pos"]
            elif "model" in state:
                ckpt_dict = state["model"]
            else:
                ckpt_dict = state

            # Strip prefixes added by DataParallel or container keys
            cleaned_dict = {}
            for k, v in ckpt_dict.items():
                if k.startswith("module."):
                    cleaned_dict[k.replace("module.", "")] = v
                elif k.startswith("model_pos."):
                    cleaned_dict[k.replace("model_pos.", "")] = v
                else:
                    cleaned_dict[k] = v

            missing, unexpected = self.dstformer.load_state_dict(cleaned_dict, strict=False)
            print(f"[MotionBERTEncoder] Loaded checkpoint: {self.CHECKPOINT.name}")
            print(f"  Successfully matched: {len(cleaned_dict) - len(unexpected)} keys")
            if missing:
                print(f"  Missing keys   : {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print(f"[MotionBERTEncoder] WARNING: checkpoint not found at {self.CHECKPOINT}")

        self.projection = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.dstformer.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self):
        for p in self.dstformer.parameters():
            p.requires_grad = True

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch
        if epoch == self.freeze_epochs:
            self._unfreeze_backbone()
            print(f"[MotionBERTEncoder] Epoch {epoch}: backbone unfrozen for fine-tuning.")

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        h36m = extract_h36m_from_openpose(x)          # (B, T, 17, 3)
        rep = self.dstformer.get_representation(h36m)  # (B, T, 17, 512)
        rep = rep.mean(dim=2)                          # Pool joints -> (B, T, 512)
        out = self.projection(rep)                     # Project -> (B, T, hidden_dim)
        return out


# ---------------------------------------------------------------------------
# Decoder (unchanged)
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, vocab_size: int, nhead: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.embedding(tgt)
        x = self.pos_encoding(x)
        x = self.transformer(
            x, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_projection(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SignLanguageTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]

        self.encoder = MotionBERTEncoder(
            hidden_dim=m["decoder_hidden_dim"],
            dropout=m.get("encoder_dropout", 0.1),
            freeze_epochs=m.get("freeze_epochs", 10),
        )
        self.decoder = Decoder(
            hidden_dim=m["decoder_hidden_dim"],
            num_layers=m["decoder_layers"],
            dropout=m["decoder_dropout"],
            vocab_size=m["vocab_size"],
        )

    def make_causal_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None) -> torch.Tensor:
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        tgt_mask = self.make_causal_mask(tgt.size(1)).to(src.device)
        return self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )