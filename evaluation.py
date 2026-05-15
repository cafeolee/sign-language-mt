import json
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
import sacrebleu

from src.data.dataset import How2SignDataset, CorpusTokenizer
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
    beam_size=5,
    length_penalty=0.7,
    repetition_penalty=1.3,
):
    model.eval()
    src = src.unsqueeze(0).to(device)
    src_key_padding_mask = src_key_padding_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        memory_single = model.encoder(src, src_key_padding_mask=src_key_padding_mask)
        # Expande memory para beam_size → (beam_size, T, hidden_dim)
        memory = memory_single.expand(beam_size, -1, -1).contiguous()
        mem_mask = src_key_padding_mask.expand(beam_size, -1).contiguous()

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    # (score, tokens, beam_idx)
    beams = [(0.0, [cls_id])]
    completed = []

    with torch.no_grad():
        for step in range(max_length):
            if not beams:
                break

            n_active = len(beams)
            # Construye batch de todos los beams activos
            tgt_batch = torch.tensor(
                [b[1] for b in beams], dtype=torch.long, device=device
            )  # (n_active, current_len)

            tgt_mask = model.make_causal_mask(tgt_batch.size(1)).to(device)

            # Usa solo las primeras n_active filas de memory
            output = model.decoder(
                tgt_batch,
                memory[:n_active],
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mem_mask[:n_active],  # ← FIX crítico
            )

            logits = output[:, -1, :]  # (n_active, vocab_size)

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
                    candidate = tokens + [next_id]
                    length_norm = (len(candidate) ** length_penalty)
                    score = (log_prob * (len(tokens) ** length_penalty) + next_lp) / length_norm
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


def evaluate(config_path: str = "configs/config.yaml"):
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_path = config["data"]["vocab_path"]
    tokenizer = CorpusTokenizer(vocab_path)

    test_dataset = How2SignDataset(
        config["data"]["processed_dir"],
        "test",
        vocab_path=vocab_path,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
        reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)

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
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    print("\n" + "=" * 50)
    print(f"BLEU  : {bleu.score:.2f}")
    print(f"chrF  : {chrf.score:.2f}")
    print("=" * 50)

    results = {
        "bleu": bleu.score,
        "chrf": chrf.score,
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