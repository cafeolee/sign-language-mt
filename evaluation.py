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


def greedy_decode(
    model,
    src,
    src_key_padding_mask,
    tokenizer,
    max_length,
    device
):
    model.eval()

    src = src.unsqueeze(0).to(device)
    src_key_padding_mask = src_key_padding_mask.unsqueeze(0).to(device)

    memory = model.encoder(
        src,
        src_key_padding_mask=src_key_padding_mask
    )

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    generated = torch.tensor([[cls_token_id]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            tgt_mask = model.make_causal_mask(generated.size(1)).to(device)

            output = model.decoder(
                generated,
                memory,
                tgt_mask=tgt_mask
            )

            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == sep_token_id:
                break

    token_ids = generated[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    text = tokenizer.convert_tokens_to_string(
        [t for t in tokens if t not in ["[CLS]", "[SEP]", "[PAD]"]]
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

        prediction = greedy_decode(
            model,
            sequence,
            src_key_padding_mask,
            tokenizer,
            config["model"]["max_output_length"],
            device,
        )

        predictions.append(prediction)
        references.append(reference_text)

        if i < 10:
            print(f"\n[{i+1}]")
            print(f"  Reference: {reference_text}")
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