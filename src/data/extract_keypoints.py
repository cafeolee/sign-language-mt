import cv2
import mediapipe as mp
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads the YAML configuration file and returns it as a dictionary."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_holistic_model(config: dict):
    """
    Initializes and returns a MediaPipe Holistic model using confidence
    thresholds defined in the config. The model extracts pose, face,
    and hand landmarks from video frames.
    """
    mp_holistic = mp.solutions.holistic
    return mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=config["keypoints"]["min_detection_confidence"],
        min_tracking_confidence=config["keypoints"]["min_tracking_confidence"],
    )


def extract_keypoints_from_frame(results) -> np.ndarray:
    """
    Extracts and concatenates pose and hand landmarks from a single frame's
    MediaPipe results into a flat feature vector of shape (258,).

    If a landmark group is not detected, its section is filled with zeros.
    """
    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        ).flatten()
    else:
        pose = np.zeros(33 * 4)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()
    else:
        left_hand = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()
    else:
        right_hand = np.zeros(21 * 3)

    return np.concatenate([pose, left_hand, right_hand])


def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """
    Normalizes a keypoint sequence so it is invariant to the signer's
    position and size in the frame.
    """
    sequence = sequence.copy()

    left_shoulder  = sequence[:, 44:46]
    right_shoulder = sequence[:, 48:50]
    mid_shoulder   = (left_shoulder + right_shoulder) / 2

    shoulder_dist = np.linalg.norm(
        left_shoulder - right_shoulder, axis=1, keepdims=True
    )
    shoulder_dist = np.where(shoulder_dist < 1e-6, 1.0, shoulder_dist)

    for t in range(len(sequence)):
        cx, cy = mid_shoulder[t]
        scale  = shoulder_dist[t, 0]
        sequence[t, 0:132:4] = (sequence[t, 0:132:4] - cx) / scale
        sequence[t, 1:132:4] = (sequence[t, 1:132:4] - cy) / scale

    return sequence


def pad_or_truncate(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """
    Ensures all sequences have the same temporal length for batching.
    Sequences longer than max_frames are truncated from the end.
    Shorter sequences are zero-padded at the end.
    """
    T = sequence.shape[0]
    if T >= max_frames:
        return sequence[:max_frames]
    pad = np.zeros((max_frames - T, sequence.shape[1]))
    return np.concatenate([sequence, pad], axis=0)


def process_video(video_path: str, holistic, max_frames: int) -> np.ndarray | None:
    """
    Runs the full keypoint extraction pipeline on a single video file:
    reads frames, extracts MediaPipe landmarks, normalizes and truncates.

    Returns:
        Array of shape (max_frames, 258), or None if the video cannot
        be opened or yields no frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_keypoints = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        frames_keypoints.append(extract_keypoints_from_frame(results))

    cap.release()

    if len(frames_keypoints) == 0:
        return None

    sequence = np.array(frames_keypoints)
    sequence = normalize_keypoints(sequence)
    sequence = pad_or_truncate(sequence, max_frames)
    return sequence


def build_video_index(wlasl_json_path: str, video_dir: str, num_classes: int) -> list[dict]:
    """
    Parses the WLASL JSON annotation file and builds a list of records
    for all video files that exist on disk, limited to the first
    num_classes glosses.

    Each record contains:
      - video_path: absolute path to the .mp4 file
      - gloss:      the sign label (e.g. "BOOK", "DRINK")
      - video_id:   unique identifier used as the .npy filename
      - split:      "train", "val", or "test"

    Args:
        wlasl_json_path: Path to WLASL_v0.3.json.
        video_dir:       Directory where downloaded .mp4 files are stored.
        num_classes:     Number of gloss classes to include.

    Returns:
        List of record dicts for videos found on disk.
    """
    with open(wlasl_json_path) as f:
        data = json.load(f)

    index = []
    for entry in data[:num_classes]:
        gloss = entry["gloss"]
        for instance in entry["instances"]:
            video_path = os.path.join(video_dir, f"{instance['video_id']}.mp4")
            if os.path.exists(video_path):
                index.append({
                    "video_path": video_path,
                    "gloss":      gloss,
                    "video_id":   instance["video_id"],
                    "split":      instance["split"],
                })
    return index


def extract_all(config_path: str = "configs/config.yaml"):
    """
    Entry point for the keypoint extraction pipeline.
    Reads the config, indexes available videos, runs MediaPipe on each one,
    and saves the resulting keypoint arrays as .npy files.

    Skips videos that have already been processed, so it is safe to
    resume if the process is interrupted.

    Saves:
      - data/keypoints/{video_id}.npy  — one file per video
      - data/splits/video_index.json   — full index with splits and labels
      - data/keypoints/failed_videos.txt (if any failures occur)
    """
    config     = load_config(config_path)
    data_cfg   = config["data"]
    max_frames = data_cfg["max_frames"]

    output_dir = Path(data_cfg["keypoints_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Indexing available videos...")
    index = build_video_index(
        wlasl_json_path=data_cfg["wlasl_json"],
        video_dir=data_cfg["raw_video_dir"],
        num_classes=data_cfg["num_classes"],
    )
    print(f"  → {len(index)} videos found on disk")

    splits_dir = Path(data_cfg["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "video_index.json", "w") as f:
        json.dump(index, f, indent=2)

    holistic = get_holistic_model(config)
    failed   = []

    for item in tqdm(index, desc="Extracting keypoints"):
        out_path = output_dir / f"{item['video_id']}.npy"
        if out_path.exists():
            continue

        sequence = process_video(item["video_path"], holistic, max_frames)
        if sequence is None:
            failed.append(item["video_id"])
            continue

        np.save(out_path, sequence)

    holistic.close()

    print(f"\nExtraction complete:")
    print(f"Processed: {len(index) - len(failed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        with open(output_dir / "failed_videos.txt", "w") as f:
            f.write("\n".join(failed))
        print(f"Failed IDs saved to {output_dir}/failed_videos.txt")


if __name__ == "__main__":
    extract_all()