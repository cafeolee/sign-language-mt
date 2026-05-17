"""
mediapipe_adapter.py
--------------------
Converts a video file to a numpy array of shape (max_frames, 274) using
MediaPipe 0.10+ Tasks API, matching the exact format produced by preprocess.py.

Output feature layout (274 values per frame):
    pose:  25 joints × 2 = 50   → indices 0:50
    face:  70 joints × 2 = 140  → indices 50:190
    left:  21 joints × 2 = 42   → indices 190:232
    right: 21 joints × 2 = 42   → indices 232:274
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions
from pathlib import Path
import urllib.request

# ---------------------------------------------------------------------------
# Download the MediaPipe Holistic task model if not present
# ---------------------------------------------------------------------------
_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
_MODEL_PATH = Path(__file__).parent / "holistic_landmarker.task"

def _ensure_model():
    if not _MODEL_PATH.exists():
        print("[mediapipe_adapter] Downloading holistic landmarker model...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[mediapipe_adapter] Model saved to {_MODEL_PATH}")

# ---------------------------------------------------------------------------
# MediaPipe pose index → OpenPose 25-joint index mapping
#
# OpenPose 25-joint layout:
#   0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist,
#   5=LShoulder, 6=LElbow, 7=LWrist, 8=MidHip,
#   9=RHip, 10=RKnee, 11=RAnkle, 12=LHip, 13=LKnee, 14=LAnkle,
#   15=REye, 16=LEye, 17=REar, 18=LEar,
#   19=LBigToe, 20=LSmallToe, 21=LHeel, 22=RBigToe, 23=RSmallToe, 24=RHeel
#
# MediaPipe 33-landmark pose layout (subset used):
#   0=Nose, 2=REye, 5=LEye, 7=REar, 8=LEar,
#   11=LShoulder, 12=RShoulder, 13=LElbow, 14=RElbow,
#   15=LWrist, 16=RWrist, 23=LHip, 24=RHip,
#   25=LKnee, 26=RKnee, 27=LAnkle, 28=RAnkle,
#   29=LHeel, 30=RHeel, 31=LFootIndex, 32=RFootIndex
#
# -1 = synthesize by averaging two landmarks
# ---------------------------------------------------------------------------
OPENPOSE_TO_MEDIAPIPE = {
    0:  0,    # Nose
    1:  -1,   # Neck       → avg(LShoulder=11, RShoulder=12)
    2:  12,   # RShoulder
    3:  14,   # RElbow
    4:  16,   # RWrist
    5:  11,   # LShoulder
    6:  13,   # LElbow
    7:  15,   # LWrist
    8:  -1,   # MidHip     → avg(LHip=23, RHip=24)
    9:  24,   # RHip
    10: 26,   # RKnee
    11: 28,   # RAnkle
    12: 23,   # LHip
    13: 25,   # LKnee
    14: 27,   # LAnkle
    15: 2,    # REye
    16: 5,    # LEye
    17: 8,    # REar
    18: 7,    # LEar
    19: 31,   # LBigToe  (LFootIndex)
    20: 31,   # LSmallToe (approx LFootIndex)
    21: 29,   # LHeel
    22: 32,   # RBigToe  (RFootIndex)
    23: 32,   # RSmallToe (approx RFootIndex)
    24: 30,   # RHeel
}

# Sample 70 face landmarks evenly from MediaPipe's 478-point face mesh
FACE_INDICES = np.linspace(0, 477, 70, dtype=int).tolist()

_NECK_PAIR   = (11, 12)  # LShoulder, RShoulder
_MIDHIP_PAIR = (23, 24)  # LHip, RHip


def _extract_pose(pose_landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    joints = np.zeros((25, 2), dtype=np.float32)
    if pose_landmarks is None:
        return joints
    lm = pose_landmarks  # ya es una lista de NormalizedLandmark
    for op_idx, mp_idx in OPENPOSE_TO_MEDIAPIPE.items():
        if mp_idx == -1:
            a, b = _NECK_PAIR if op_idx == 1 else _MIDHIP_PAIR
            joints[op_idx, 0] = (lm[a].x + lm[b].x) / 2 * frame_w
            joints[op_idx, 1] = (lm[a].y + lm[b].y) / 2 * frame_h
        else:
            joints[op_idx, 0] = lm[mp_idx].x * frame_w
            joints[op_idx, 1] = lm[mp_idx].y * frame_h
    return joints


def _extract_face(face_landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    joints = np.zeros((70, 2), dtype=np.float32)
    if face_landmarks is None:
        return joints
    lm = face_landmarks
    for out_idx, mp_idx in enumerate(FACE_INDICES):
        joints[out_idx, 0] = lm[mp_idx].x * frame_w
        joints[out_idx, 1] = lm[mp_idx].y * frame_h
    return joints


def _extract_hand(hand_landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    joints = np.zeros((21, 2), dtype=np.float32)
    if hand_landmarks is None:
        return joints
    for i, lm_point in enumerate(hand_landmarks):
        joints[i, 0] = lm_point.x * frame_w
        joints[i, 1] = lm_point.y * frame_h
    return joints

def _normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Applies the same global normalization as preprocess.py:
    centers on shoulder midpoint, scales by inter-shoulder distance,
    using median statistics from valid frames only.
    """
    sequence = sequence.copy()
    T        = sequence.shape[0]

    # RShoulder = OpenPose kp2 → flat indices 4,5
    # LShoulder = OpenPose kp5 → flat indices 10,11
    right_shoulder = sequence[:, 4:6]
    left_shoulder  = sequence[:, 10:12]
    mid_shoulder   = (left_shoulder + right_shoulder) / 2
    shoulder_dist  = np.linalg.norm(left_shoulder - right_shoulder, axis=1)

    valid_mask = shoulder_dist > 1e-6

    if valid_mask.sum() == 0:
        return np.zeros((T, 274), dtype=np.float32)

    global_scale = np.median(shoulder_dist[valid_mask])
    global_cx    = np.median(mid_shoulder[valid_mask, 0])
    global_cy    = np.median(mid_shoulder[valid_mask, 1])

    x_indices = np.arange(0, 274, 2)
    y_indices = np.arange(1, 274, 2)

    sequence[:, x_indices] = (sequence[:, x_indices] - global_cx) / global_scale
    sequence[:, y_indices] = (sequence[:, y_indices] - global_cy) / global_scale
    sequence[~valid_mask]  = 0.0

    return sequence.astype(np.float32)


def _same_length(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """Truncates or zero-pads sequence to exactly max_frames."""
    T = sequence.shape[0]
    if T >= max_frames:
        return sequence[:max_frames]
    pad = np.zeros((max_frames - T, sequence.shape[1]), dtype=np.float32)
    return np.concatenate([sequence, pad], axis=0)


def video_to_keypoints(
    video_path: str,
    max_frames: int = 150,
) -> np.ndarray:
    """
    Converts a video file to a normalized keypoint array compatible with
    the SignLanguageTransformer model.

    Args:
        video_path: Path to the input video file.
        max_frames: Maximum number of frames (must match model config).

    Returns:
        np.ndarray of shape (max_frames, 274), or None if video cannot be opened.
    """
    _ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0

    base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
    options      = HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    frames    = []
    frame_idx = 0

    with HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            pose_lm  = result.pose_landmarks       if result.pose_landmarks       else None
            face_lm  = result.face_landmarks       if result.face_landmarks       else None
            left_lm  = result.left_hand_landmarks  if result.left_hand_landmarks  else None
            right_lm = result.right_hand_landmarks if result.right_hand_landmarks else None

            pose  = _extract_pose(pose_lm,  frame_w, frame_h)
            face  = _extract_face(face_lm,  frame_w, frame_h)
            left  = _extract_hand(left_lm,  frame_w, frame_h)
            right = _extract_hand(right_lm, frame_w, frame_h)

            frame_features = np.concatenate([
                pose.flatten(),   # 50
                face.flatten(),   # 140
                left.flatten(),   # 42
                right.flatten(),  # 42
            ])

            frames.append(frame_features)
            frame_idx += 1

    cap.release()

    if not frames:
        return None

    sequence = np.stack(frames, axis=0)
    sequence = _normalize_sequence(sequence)
    sequence = _same_length(sequence, max_frames)

    return sequence