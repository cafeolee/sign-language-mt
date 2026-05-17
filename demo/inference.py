"""
Loads the SignLanguageTransformer model and runs beam search decoding
on a preprocessed keypoint sequence.

Designed to be imported by app.py and called once per user request.
The model is loaded once at startup and reused across requests.
"""
import sys
import torch
import yaml
import numpy as np
from pathlib import Path

from src.data.dataset import CorpusTokenizer
from src.model.model import SignLanguageTransformer



# Default paths
sys.path.insert(0, str(Path(__file__).parent.parent))

CONFIG_PATH = Path(__file__).parent.parent / "configs/config.yaml"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints/best_model.pt"
VOCAB_PATH = Path(__file__).parent.parent / "data/vocab.json"


def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(
    config_path: str = CONFIG_PATH,
    checkpoint_path: str = CHECKPOINT_PATH,
) -> tuple:
    """
    Loads the model and tokenizer from disk.
    Returns (model, tokenizer, config, device).
    Call this once at app startup, not per request.
    """
    config    = load_config(config_path)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CorpusTokenizer(config["data"]["vocab_path"])

    model = SignLanguageTransformer(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"[inference] Model loaded from {checkpoint_path} on {device}")
    return model, tokenizer, config, device


def predict(
    sequence: np.ndarray,
    model: SignLanguageTransformer,
    tokenizer: CorpusTokenizer,
    config: dict,
    device: torch.device,
    beam_size: int = 5,
    length_penalty: float = 0.7,
    repetition_penalty: float = 2.5,
) -> str:
    """
    Runs beam search decoding on a preprocessed keypoint sequence.

    Args:
        sequence:   np.ndarray of shape (max_frames, 274), normalized.
        model:      Loaded SignLanguageTransformer instance.
        tokenizer:  CorpusTokenizer instance.
        config:     Config dict loaded from config.yaml.
        device:     torch.device to run inference on.
        beam_size:          Number of beams for beam search.
        length_penalty:     Penalizes short sequences (< 1.0 favors shorter).
        repetition_penalty: Penalizes repeated tokens (> 1.0 reduces repetition).

    Returns:
        Predicted translation as a string.
    """
    max_length = config["model"]["max_output_length"]

    src = torch.tensor(sequence, dtype=torch.float32).to(device)  # (T, 274)
    valid_frames         = (src.abs().sum(dim=1) > 0)             # (T,)
    src_key_padding_mask = (~valid_frames).unsqueeze(0).to(device) # (1, T)
    src                  = src.unsqueeze(0)                        # (1, T, 274)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    with torch.no_grad():
        memory_single = model.encoder(src, src_key_padding_mask=src_key_padding_mask)
        memory   = memory_single.expand(beam_size, -1, -1).contiguous()
        mem_mask = src_key_padding_mask.expand(beam_size, -1).contiguous()

    beams     = [(0.0, [cls_id])]
    completed = []

    with torch.no_grad():
        for _ in range(max_length):
            if not beams:
                break

            n_active  = len(beams)
            tgt_batch = torch.tensor(
                [b[1] for b in beams], dtype=torch.long, device=device
            )
            tgt_mask = model.make_causal_mask(tgt_batch.size(1)).to(device)

            output = model.decoder(
                tgt_batch,
                memory[:n_active],
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mem_mask[:n_active],
            )

            logits = output[:, -1, :]

            all_candidates = []
            for i, (log_prob, tokens) in enumerate(beams):
                token_logits = logits[i].clone()

                # Repetition penalty
                for token_id in set(tokens):
                    if token_logits[token_id] > 0:
                        token_logits[token_id] /= repetition_penalty
                    else:
                        token_logits[token_id] *= repetition_penalty

                lp = torch.log_softmax(token_logits, dim=-1)
                topk_lp, topk_ids = lp.topk(beam_size)

                for next_lp, next_id in zip(topk_lp.tolist(), topk_ids.tolist()):
                    candidate   = tokens + [next_id]
                    length_norm = len(candidate) ** length_penalty
                    score       = (log_prob * (len(tokens) ** length_penalty) + next_lp) / length_norm
                    all_candidates.append((score, candidate))

            all_candidates.sort(key=lambda x: x[0], reverse=True)

            beams = []
            for score, tokens in all_candidates:
                if tokens[-1] == sep_id:
                    completed.append((score, tokens))
                else:
                    beams.append((score, tokens))
                if len(beams) >= beam_size:
                    break

            if len(completed) >= beam_size:
                break

    if not completed:
        completed = beams if beams else [(0.0, [cls_id])]

    completed.sort(key=lambda x: x[0], reverse=True)
    return tokenizer.decode(completed[0][1], skip_special_tokens=True)