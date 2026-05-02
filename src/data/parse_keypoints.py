import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from tqdm import tqdm


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


def parse_clip(clip_dir: Path) -> np.ndarray:
    json_files = sorted(clip_dir.glob("*_keypoints.json"))
    
    if not json_files:
        return None
    
    frames = [parse_keypoints_file(f) for f in json_files]
    return np.array(frames)


def same_length(sequence: np.ndarray, max_frames: int) -> np.ndarray: # truncate or pad
    T = sequence.shape[0]
    if T >= max_frames:
        return sequence[:max_frames]
    pad = np.zeros((max_frames - T, sequence.shape[1]))
    return np.concatenate([sequence, pad], axis=0)


def normalize_keypoints(sequence: np.ndarray) -> np.ndarray:
    """Normalizes pose coordinates relative to shoulder midpoint and distance"""
    sequence = sequence.copy()

    left_shoulder  = sequence[:, 6:8]
    right_shoulder = sequence[:, 15:17]
    mid_shoulder   = (left_shoulder + right_shoulder) / 2

    shoulder_dist = np.linalg.norm(
        left_shoulder - right_shoulder, axis=1, keepdims=True
    )
    shoulder_dist = np.where(shoulder_dist < 1e-6, 1.0, shoulder_dist)

    for t in range(len(sequence)):
        cx, cy = mid_shoulder[t]
        scale  = shoulder_dist[t, 0]
        sequence[t, 0:75:3] = (sequence[t, 0:75:3] - cx) / scale
        sequence[t, 1:75:3] = (sequence[t, 1:75:3] - cy) / scale

    return sequence


def build_index(csv_path: str, keypoints_dir: str, max_sentence_words: int) -> list[dict]:
    """Reads the CSV and returns a list of records linking each clip to its translation, filtered by sentence length."""
    df = pd.read_csv(csv_path, delimiter='\t')
    df = df[df['SENTENCE'].apply(lambda s: len(s.split()) <= max_sentence_words)]
    
    index = []
    for _, row in df.iterrows():
        clip_dir = Path(keypoints_dir) / row['SENTENCE_NAME']
        if clip_dir.exists():
            index.append({
                'clip_id':    row['SENTENCE_NAME'],
                'clip_dir':   str(clip_dir),
                'sentence':   row['SENTENCE'],
            })
    return index


def process_split(split: str, config: dict):
    """Processes all clips for a given split (train/val/test) and saves each as a .npy file."""
    data_cfg = config['data']
    
    index = build_index(
        csv_path         = data_cfg['translations_dir'][split],
        keypoints_dir    = data_cfg['keypoints_dir'][split],
        max_sentence_words = data_cfg['max_sentence_words'],
    )
    
    output_dir = Path(data_cfg['processed_dir']) / split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    failed = []
    for item in tqdm(index, desc=f"Processing {split}"):
        out_path = output_dir / f"{item['clip_id']}.npy"
        if out_path.exists():
            continue
        
        sequence = parse_clip(Path(item['clip_dir']))
        if sequence is None:
            failed.append(item['clip_id'])
            continue
        
        sequence = normalize_keypoints(sequence)
        sequence = same_length(sequence, data_cfg['max_frames'])
        np.save(out_path, sequence)
    
    index_path = output_dir / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"{split}: {len(index) - len(failed)} processed, {len(failed)} failed")


def main(config_path: str = 'configs/config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    for split in ['train', 'val', 'test']:
        process_split(split, config)

if __name__ == '__main__':
    main()