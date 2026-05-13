import json
import torch
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import sacrebleu
from jiwer import wer

from src.data.dataset import How2SignDataset
from src.model.model import SignLanguageTransformer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def beam_search_decode(
    model,
    src,
    src_key_padding_mask,
    tokenizer,
    max_length,
    device,
    beam_size: int = 5,
    length_penalty: float = 0.7,
    repetition_penalty: float = 1.3,
):
    """
    Beam search decoder with length penalty and repetition penalty.

    Args:
        beam_size:          Number of beams to explore in parallel.
        length_penalty:     Alpha for length normalization. Values < 1 favor
                            shorter sequences; > 1 favor longer ones. 0.7 is
                            a good starting point for short sentences.
        repetition_penalty: Penalizes tokens that have already appeared.
                            1.0 = no penalty; > 1.0 discourages repetition.
    """
    model.eval()

    src = src.unsqueeze(0).to(device)
    src_key_padding_mask = src_key_padding_mask.unsqueeze(0).to(device)

    # Encode once — reuse for all beams
    with torch.no_grad():
        memory = model.encoder(src, src_key_padding_mask=src_key_padding_mask)

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    # Each beam: (log_prob, token_ids_list)
    beams = [(0.0, [cls_token_id])]
    completed = []

    with torch.no_grad():
        for _ in range(max_length):
            all_candidates = []

            for log_prob, tokens in beams:
                # If this beam already ended, carry it forward unchanged
                if tokens[-1] == sep_token_id:
                    all_candidates.append((log_prob, tokens))
                    continue

                tgt = torch.tensor([tokens], dtype=torch.long).to(device)
                tgt_mask = model.make_causal_mask(tgt.size(1)).to(device)

                # Expand memory to match beam batch size (here always 1 — we
                # loop beams in Python, so memory stays [1, T, D])
                output = model.decoder(tgt, memory, tgt_mask=tgt_mask)

                # Logits for the next token
                logits = output[0, -1, :]  # [vocab_size]

                # --- Repetition penalty ---
                for token_id in set(tokens):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty

                log_probs = torch.log_softmax(logits, dim=-1)

                # Take top beam_size next tokens
                topk_log_probs, topk_ids = log_probs.topk(beam_size)

                for next_log_prob, next_id in zip(
                    topk_log_probs.tolist(), topk_ids.tolist()
                ):
                    candidate_tokens = tokens + [next_id]

                    # --- Length-normalized score ---
                    # score = sum_log_probs / len ^ alpha
                    length = len(candidate_tokens)
                    normalized_score = (log_prob + next_log_prob) / (length ** length_penalty)

                    all_candidates.append((normalized_score, candidate_tokens))

            # Keep only the top beam_size candidates
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_size]

            # Move finished beams to completed
            still_running = []
            for score, tokens in beams:
                if tokens[-1] == sep_token_id:
                    completed.append((score, tokens))
                else:
                    still_running.append((score, tokens))

            beams = still_running

            # Stop early if we have enough completed beams
            if len(completed) >= beam_size:
                break

        # If nothing completed, use whatever is still running
        if not completed:
            completed = beams

    # Pick the highest-scoring completed sequence
    completed.sort(key=lambda x: x[0], reverse=True)
    best_tokens = completed[0][1]

    # Decode to text
    tokens_str = tokenizer.convert_ids_to_tokens(best_tokens)
    text = tokenizer.convert_tokens_to_string(
        [t for t in tokens_str if t not in ["[CLS]", "[SEP]", "[PAD]"]]
    )
    return text.strip()


def evaluate(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_dataset = How2SignDataset(
        config["data"]["processed_dir"],
        "test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    model = SignLanguageTransformer(config).to(device)

    checkpoint_path = Path("checkpoints") / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    predictions = []
    references = []

    print("\nGenerating predictions...")

    for i, (sequence, labels, attention_mask, valid_frames) in enumerate(test_loader):

        sequence = sequence.squeeze(0).to(device)
        valid_frames = valid_frames.squeeze(0)
        src_key_padding_mask = (~valid_frames).to(device)

        reference_ids = labels[0].tolist()
        reference_tokens = tokenizer.convert_ids_to_tokens(reference_ids)
        reference_text = tokenizer.convert_tokens_to_string(
            [t for t in reference_tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]]
        ).strip()

        prediction = beam_search_decode(
            model,
            sequence,
            src_key_padding_mask,
            tokenizer,
            config["model"]["max_output_length"],
            device,
            beam_size=5,
            length_penalty=0.7,
            repetition_penalty=1.3,
        )

        predictions.append(prediction)
        references.append(reference_text)

        if i < 10:
            print(f"\n[{i+1}]")
            print(f"  Reference:  {reference_text}")
            print(f"  Prediction: {prediction}")

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    word_error_rate = wer(references, predictions)

    print("\n" + "=" * 50)
    print(f"BLEU score : {bleu.score:.2f}")
    print(f"WER        : {word_error_rate:.4f}")
    print("=" * 50)

    results = {
        "bleu": bleu.score,
        "wer": word_error_rate,
        "beam_size": 5,
        "length_penalty": 0.7,
        "repetition_penalty": 1.3,
        "examples": [
            {"reference": r, "prediction": p}
            for r, p in zip(references[:20], predictions[:20])
        ],
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nFull results saved to evaluation_results.json")


if __name__ == "__main__":
    evaluate()