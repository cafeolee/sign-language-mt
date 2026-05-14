import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm

# OpenPose layout (each keypoint = x, y, confidence):
#   pose:  25 keypoints × 3 = 75  values  → indices 0:75
#   face:  70 keypoints × 3 = 210 values  → indices 75:285
#   left:  21 keypoints × 3 = 63  values  → indices 285:348
#   right: 21 keypoints × 3 = 63  values  → indices 348:411
#
# After dropping confidence scores the output is:
#   pose:  25 × 2 = 50
#   face:  70 × 2 = 140
#   left:  21 × 2 = 42
#   right: 21 × 2 = 42
#   TOTAL: 274 features  ← update feature_dim in config.yaml


def parse_keypoints_file(json_path: str) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)

    if not data["people"]:
        return np.zeros(411)

    person = data["people"][0]

    pose  = np.array(person["pose_keypoints_2d"])
    face  = np.array(person["face_keypoints_2d"])
    left  = np.array(person["hand_left_keypoints_2d"])
    right = np.array(person["hand_right_keypoints_2d"])

    return np.concatenate([pose, face, left, right])


def drop_confidence_scores(frame: np.ndarray) -> np.ndarray:
    """
    OpenPose stores keypoints as [x, y, confidence, x, y, confidence, ...].
    Drop every third value (confidence) and keep only (x, y) pairs.
    411 values → 274 values.
    """
    return np.delete(frame, np.arange(2, frame.shape[0], 3))


def parse_clip(clip_dir: Path) -> np.ndarray | None:
    json_files = sorted(clip_dir.glob("*_keypoints.json"))

    if not json_files:
        return None

    frames = [parse_keypoints_file(f) for f in json_files]
    return np.array(frames)  # (T, 411)


def same_length(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """Truncate or pad sequence to exactly max_frames."""
    T = sequence.shape[0]
    if T >= max_frames:
        return sequence[:max_frames]
    pad = np.zeros((max_frames - T, sequence.shape[1]))
    return np.concatenate([sequence, pad], axis=0)


def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """
    Normalizes ALL keypoints (pose, face, hands) relative to shoulder midpoint
    and inter-shoulder distance, then drops confidence scores.

    Steps:
      1. Compute shoulder midpoint and scale from body keypoints.
      2. Apply the same centering + scaling to every (x, y) pair across all parts.
      3. Drop confidence scores → output shape (T, 274).
    """
    sequence = sequence.copy()
    T = len(sequence)

    # --- Shoulder reference (from body keypoints, before dropping confidences) ---
    # OpenPose body: keypoint 2 = right shoulder (idx 6,7), keypoint 5 = left shoulder (idx 15,16)
    left_shoulder  = sequence[:, 15:17]   # (T, 2)
    right_shoulder = sequence[:, 6:8]     # (T, 2)
    mid_shoulder   = (left_shoulder + right_shoulder) / 2  # (T, 2)

    shoulder_dist = np.linalg.norm(
        left_shoulder - right_shoulder, axis=1, keepdims=True
    )  # (T, 1)
    # Avoid division by zero for frames where shoulders weren't detected
    shoulder_dist = np.where(shoulder_dist < 1e-6, 1.0, shoulder_dist)

    # --- Normalize every (x, y) pair across all 137 keypoints ---
    # Total keypoints: 25 body + 70 face + 21 left + 21 right = 137
    # In the raw 411-value layout, x values are at indices 0,3,6,... and y at 1,4,7,...
    x_indices = np.arange(0, 411, 3)  # 137 x-coordinates
    y_indices = np.arange(1, 411, 3)  # 137 y-coordinates

    cx = mid_shoulder[:, 0]   # (T,)
    cy = mid_shoulder[:, 1]   # (T,)
    scale = shoulder_dist[:, 0]  # (T,)

    sequence[:, x_indices] = (sequence[:, x_indices] - cx[:, None]) / scale[:, None]
    sequence[:, y_indices] = (sequence[:, y_indices] - cy[:, None]) / scale[:, None]

    # --- Drop confidence scores ---
    # Apply per-frame and stack back
    frames_xy = np.stack([drop_confidence_scores(sequence[t]) for t in range(T)])
    # Shape: (T, 274)

    return frames_xy


def build_index(
    csv_path: str,
    keypoints_dir: str,
    max_sentence_words: int,
) -> list[dict]:
    """Reads the CSV and returns records linking each clip to its translation."""
    df = pd.read_csv(csv_path, delimiter="\t")
    df = df[df["SENTENCE"].apply(lambda s: len(s.split()) <= max_sentence_words)]

    index = []
    for _, row in df.iterrows():
        clip_dir = Path(keypoints_dir) / row["SENTENCE_NAME"]
        if clip_dir.exists():
            index.append({
                "clip_id":  row["SENTENCE_NAME"],
                "clip_dir": str(clip_dir),
                "sentence": row["SENTENCE"],
            })
    return index


def process_split(split: str, config: dict):
    """Processes all clips for a given split and saves each as a .npy file."""
    data_cfg = config["data"]

    index = build_index(
        csv_path           = data_cfg["translations_dir"][split],
        keypoints_dir      = data_cfg["keypoints_dir"][split],
        max_sentence_words = data_cfg["max_sentence_words"],
    )

    output_dir = Path(data_cfg["processed_dir"]) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    failed = []
    for item in tqdm(index, desc=f"Processing {split}"):
        out_path = output_dir / f"{item['clip_id']}.npy"
        if out_path.exists():
            continue

        sequence = parse_clip(Path(item["clip_dir"]))
        if sequence is None:
            failed.append(item["clip_id"])
            continue

        sequence = normalize_keypoints(sequence)   # (T, 274) normalized
        sequence = same_length(sequence, data_cfg["max_frames"])
        np.save(out_path, sequence)

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"{split}: {len(index) - len(failed)} processed, {len(failed)} failed")


def main(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for split in ["train", "val", "test"]:
        process_split(split, config)


if __name__ == "__main__":
    main()