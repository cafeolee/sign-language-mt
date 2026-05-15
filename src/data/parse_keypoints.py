import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# OpenPose raw layout (each keypoint = x, y, confidence):
#   pose:  25 keypoints × 3 = 75  values  → indices 0:75
#   face:  70 keypoints × 3 = 210 values  → indices 75:285
#   left:  21 keypoints × 3 = 63  values  → indices 285:348
#   right: 21 keypoints × 3 = 63  values  → indices 348:411
#
# After dropping confidence scores the output layout is:
#   pose:  25 × 2 = 50   → indices 0:50
#   face:  70 × 2 = 140  → indices 50:190
#   left:  21 × 2 = 42   → indices 190:232
#   right: 21 × 2 = 42   → indices 232:274
#   TOTAL: 274 features
# ---------------------------------------------------------------------------


def parse_keypoints_file(json_path: str) -> np.ndarray:
    """
    Parses a single OpenPose JSON file and returns a flat array of shape (411,).
    Returns zeros if no person was detected in the frame.
    """
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
    Drops every third value (confidence) and keeps only (x, y) pairs.
    Input shape: (411,) → Output shape: (274,)
    """
    return np.delete(frame, np.arange(2, frame.shape[0], 3))


def parse_clip(clip_dir: Path) -> np.ndarray | None:
    """
    Loads all keypoint JSON files for a clip and stacks them into (T, 411).
    Returns None if no JSON files are found.
    """
    json_files = sorted(clip_dir.glob("*_keypoints.json"))

    if not json_files:
        return None

    frames = [parse_keypoints_file(f) for f in json_files]
    return np.array(frames)  # (T, 411)


def same_length(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Truncates or zero-pads a sequence to exactly max_frames along the time axis.
    Padding is added at the end so valid frames always come first.
    """
    T = sequence.shape[0]
    if T >= max_frames:
        return sequence[:max_frames]
    pad = np.zeros((max_frames - T, sequence.shape[1]))
    return np.concatenate([sequence, pad], axis=0)


def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """
    Normalizes all keypoints (pose, face, hands) using a single global reference
    computed from the valid frames of the clip, rather than per-frame statistics.

    Using per-frame normalization causes inconsistent scales across frames when
    detection confidence varies, which breaks MotionBERT's motion modeling.

    Steps:
      1. Identify valid frames (those where both shoulders were detected).
      2. Compute a global center and scale from the median of valid frames.
      3. Apply the same centering + scaling to every (x, y) pair in all frames.
      4. Zero out frames where no person was detected (instead of leaving
         them as normalized noise).
      5. Drop confidence scores → output shape (T, 274).

    Reference keypoints (before dropping confidence):
      OpenPose body keypoint 2 = right shoulder → raw indices 6, 7
      OpenPose body keypoint 5 = left  shoulder → raw indices 15, 16
    """
    sequence = sequence.copy()
    T = len(sequence)

    left_shoulder  = sequence[:, 15:17]
    right_shoulder = sequence[:, 6:8]
    mid_shoulder   = (left_shoulder + right_shoulder) / 2
    shoulder_dist  = np.linalg.norm(left_shoulder - right_shoulder, axis=1)

    valid_mask = shoulder_dist > 1e-6

    # Check BEFORE calling np.median to avoid empty slice warnings
    if valid_mask.sum() == 0:
        return np.zeros((T, 274))

    global_scale = np.median(shoulder_dist[valid_mask])
    global_cx    = np.median(mid_shoulder[valid_mask, 0])
    global_cy    = np.median(mid_shoulder[valid_mask, 1])

    # Normalize all (x, y) pairs using the global reference
    x_indices = np.arange(0, 411, 3)  # 137 x-coordinates
    y_indices = np.arange(1, 411, 3)  # 137 y-coordinates

    sequence[:, x_indices] = (sequence[:, x_indices] - global_cx) / global_scale
    sequence[:, y_indices] = (sequence[:, y_indices] - global_cy) / global_scale

    # Explicitly zero out frames with no detection instead of leaving
    # them as (0 - global_cx) / global_scale, which would be non-zero noise
    sequence[~valid_mask] = 0.0

    # Drop confidence scores for all frames
    frames_xy = np.stack([drop_confidence_scores(sequence[t]) for t in range(T)])
    # Shape: (T, 274)

    return frames_xy


def build_index(
    csv_path: str,
    keypoints_dir: str,
    max_sentence_words: int,
) -> list[dict]:
    """
    Reads the How2Sign CSV and returns records linking each clip to its translation.
    Filters out sentences longer than max_sentence_words and clips whose
    keypoint directory does not exist on disk.
    """
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
    """
    Processes all clips for a given split:
      1. Parses raw OpenPose JSON files.
      2. Normalizes keypoints with global per-clip statistics.
      3. Pads/truncates to max_frames.
      4. Saves each clip as a .npy file.
      5. Writes an index.json mapping clip_id → sentence.

    Already-processed clips are skipped to allow resuming interrupted runs.
    """
    data_cfg = config["data"]

    index = build_index(
        csv_path           = data_cfg["translations_dir"][split],
        keypoints_dir      = data_cfg["keypoints_dir"][split],
        max_sentence_words = data_cfg["max_sentence_words"],
    )

    output_dir = Path(data_cfg["processed_dir"]) / split
    output_dir.mkdir(parents=True, exist_ok=True)

    failed  = []
    skipped = 0

    for item in tqdm(index, desc=f"Processing {split}"):
        out_path = output_dir / f"{item['clip_id']}.npy"

        if out_path.exists():
            skipped += 1
            continue

        sequence = parse_clip(Path(item["clip_dir"]))
        if sequence is None:
            failed.append(item["clip_id"])
            continue

        sequence = normalize_keypoints(sequence)              # (T, 274) normalized
        sequence = same_length(sequence, data_cfg["max_frames"])  # pad/truncate to max_frames
        np.save(out_path, sequence.astype(np.float32))

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    processed = len(index) - len(failed) - skipped
    print(f"{split}: {processed} processed, {skipped} skipped (already existed), {len(failed)} failed")
    if failed:
        print(f"  Failed clip IDs: {failed[:5]}{'...' if len(failed) > 5 else ''}")


def main(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for split in ["train", "val", "test"]:
        process_split(split, config)


if __name__ == "__main__":
    main()