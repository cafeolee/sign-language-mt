"""
Converts a video file to a numpy array of shape (max_frames, 274) using
MediaPipe, matching the exact format produced by preprocess.py with OpenPose.

Output feature layout (274 values per frame, confidence scores dropped):
    pose:  25 joints × 2 = 50   → indices 0:50
    face:  70 joints × 2 = 140  → indices 50:190
    left:  21 joints × 2 = 42   → indices 190:232
    right: 21 joints × 2 = 42   → indices 232:274

MediaPipe Holistic provides:
    - 33 pose landmarks (we map 25 of them to match OpenPose body layout)
    - 468 face landmarks (we sample 70 to match OpenPose face count)
    - 21 left hand landmarks
    - 21 right hand landmarks
"""

import cv2
import numpy as np
import mediapipe as mp

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
# MediaPipe 33-landmark layout (subset used):
#   0=Nose, 11=LShoulder, 12=RShoulder, 13=LElbow, 14=RElbow,
#   15=LWrist, 16=RWrist, 23=LHip, 24=RHip, 25=LKnee, 26=RKnee,
#   27=LAnkle, 28=RAnkle, 29=LHeel, 30=RHeel, 31=LFootIndex, 32=RFootIndex,
#   3=LMouth(ear approx), 4=RMouth(ear approx), 5=LEye, 2=REye
# ---------------------------------------------------------------------------

# Maps OpenPose joint index → MediaPipe landmark index
# -1 means synthesize from two landmarks (averaged)
OPENPOSE_TO_MEDIAPIPE = {
    0:  0,    # Nose
    1:  -1,   # Neck → average of LShoulder(11) and RShoulder(12)
    2:  12,   # RShoulder
    3:  14,   # RElbow
    4:  16,   # RWrist
    5:  11,   # LShoulder
    6:  13,   # LElbow
    7:  15,   # LWrist
    8:  -1,   # MidHip → average of LHip(23) and RHip(24)
    9:  24,   # RHip
    10: 26,   # RKnee
    11: 28,   # RAnkle
    12: 23,   # LHip
    13: 25,   # LKnee
    14: 27,   # LAnkle
    15: 2,    # REye
    16: 5,    # LEye
    17: 8,    # REar (approximated)
    18: 7,    # LEar (approximated)
    19: 31,   # LBigToe (LFootIndex)
    20: 31,   # LSmallToe (approximated with LFootIndex)
    21: 29,   # LHeel
    22: 32,   # RBigToe (RFootIndex)
    23: 32,   # RSmallToe (approximated with RFootIndex)
    24: 30,   # RHeel
}

# Face landmark indices sampled from MediaPipe 468 to match OpenPose 70
# Evenly spaced across the face mesh for coverage
FACE_INDICES = np.linspace(0, 467, 70, dtype=int).tolist()


def _extract_pose(landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Extracts 25 OpenPose-compatible body joints from MediaPipe pose landmarks.
    Returns shape (25, 2) in pixel coordinates.
    """
    joints = np.zeros((25, 2), dtype=np.float32)

    if landmarks is None:
        return joints

    lm = landmarks.landmark

    for op_idx, mp_idx in OPENPOSE_TO_MEDIAPIPE.items():
        if mp_idx == -1:
            # Synthesize Neck (op=1) and MidHip (op=8) by averaging
            if op_idx == 1:
                l = lm[11]
                r = lm[12]
            else:  # op_idx == 8
                l = lm[23]
                r = lm[24]
            joints[op_idx, 0] = (l.x + r.x) / 2 * frame_w
            joints[op_idx, 1] = (l.y + r.y) / 2 * frame_h
        else:
            joints[op_idx, 0] = lm[mp_idx].x * frame_w
            joints[op_idx, 1] = lm[mp_idx].y * frame_h

    return joints  # (25, 2)


def _extract_face(landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Samples 70 face landmarks from MediaPipe's 468-point face mesh.
    Returns shape (70, 2) in pixel coordinates.
    """
    joints = np.zeros((70, 2), dtype=np.float32)

    if landmarks is None:
        return joints

    lm = landmarks.landmark
    for out_idx, mp_idx in enumerate(FACE_INDICES):
        joints[out_idx, 0] = lm[mp_idx].x * frame_w
        joints[out_idx, 1] = lm[mp_idx].y * frame_h

    return joints  # (70, 2)


def _extract_hand(landmarks, frame_w: int, frame_h: int) -> np.ndarray:
    """
    Extracts 21 hand landmarks from MediaPipe hand landmarks.
    Returns shape (21, 2) in pixel coordinates.
    """
    joints = np.zeros((21, 2), dtype=np.float32)

    if landmarks is None:
        return joints

    for i, lm in enumerate(landmarks.landmark):
        joints[i, 0] = lm.x * frame_w
        joints[i, 1] = lm.y * frame_h

    return joints  # (21, 2)


def _normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Applies the same global normalization as preprocess.py
    """
    sequence = sequence.copy()
    T = sequence.shape[0]

    # Shoulder positions are at OpenPose body indices 2 and 5
    # In the 274-feature layout (no confidence), body is at 0:50
    # joint j → x at j*2, y at j*2+1
    right_shoulder = sequence[:, 2*2:2*2+2]   # OpenPose kp2 (RShoulder)
    left_shoulder  = sequence[:, 5*2:5*2+2]   # OpenPose kp5 (LShoulder)
    mid_shoulder   = (left_shoulder + right_shoulder) / 2
    shoulder_dist  = np.linalg.norm(left_shoulder - right_shoulder, axis=1)

    valid_mask = shoulder_dist > 1e-6

    if valid_mask.sum() == 0:
        return np.zeros((T, 274), dtype=np.float32)

    global_scale = np.median(shoulder_dist[valid_mask])
    global_cx    = np.median(mid_shoulder[valid_mask, 0])
    global_cy    = np.median(mid_shoulder[valid_mask, 1])

    # Normalize all x and y coordinates
    x_indices = np.arange(0, 274, 2)  # all x coordinates
    y_indices = np.arange(1, 274, 2)  # all y coordinates

    sequence[:, x_indices] = (sequence[:, x_indices] - global_cx) / global_scale
    sequence[:, y_indices] = (sequence[:, y_indices] - global_cy) / global_scale

    # Zero out frames with no valid detection
    sequence[~valid_mask] = 0.0

    return sequence.astype(np.float32)


def _same_length(sequence: np.ndarray, max_frames: int) -> np.ndarray:
    """Truncates or zero-pads sequence to exactly max_frames."""
    T = sequence.shape[0]
    if T >= max_frames:
        return sequence[:max_frames]
    pad = np.zeros((max_frames - T, sequence.shape[1]), dtype=np.float32)
    return np.concatenate([sequence, pad], axis=0)


def video_to_keypoints(video_path: str, max_frames: int = 150, model_complexity: int = 1) -> np.ndarray:
    """
    Converts a video file to a normalized keypoint array compatible with
    the SignLanguageTransformer model.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize MediaPipe Holistic (pose + face + both hands in one pass)
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        pose = _extract_pose(results.pose_landmarks, frame_w, frame_h)  # (25, 2)
        face = _extract_face(results.face_landmarks, frame_w, frame_h)  # (70, 2)
        left = _extract_hand(results.left_hand_landmarks, frame_w, frame_h)  # (21, 2)
        right = _extract_hand(results.right_hand_landmarks, frame_w, frame_h)  # (21, 2)

        # Flatten and concatenate → (274,)
        frame_features = np.concatenate([
            pose.flatten(), # 50
            face.flatten(), # 140
            left.flatten(), # 42
            right.flatten(), # 42
        ])
        frames.append(frame_features)

    cap.release()
    holistic.close()

    if not frames:
        return None

    sequence = np.stack(frames, axis=0) # (T, 274)
    sequence = _normalize_sequence(sequence) # normalize like preprocess.py
    sequence = _same_length(sequence, max_frames) # pad/truncate

    return sequence  # (max_frames, 274)