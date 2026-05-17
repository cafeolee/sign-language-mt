"""
Generates an animated skeleton GIF from a .npy keypoint sequence.
Used in the demo Tab 1 to visualize what the model sees when it processes
a sign language clip.

The animation shows the 25 OpenPose-compatible body joints and the
21+21 hand joints extracted during preprocessing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import tempfile



# Skeleton connectivity: pairs of joint indices to draw bones
# Based on OpenPose 25-joint body layout (indices into the 50 body features)

BODY_CONNECTIONS = [
    (0, 1), # Nose → Neck
    (1, 2), # Neck → RShoulder
    (2, 3), # RShoulder → RElbow
    (3, 4), # RElbow → RWrist
    (1, 5), # Neck → LShoulder
    (5, 6), # LShoulder → LElbow
    (6, 7), # LElbow → LWrist
    (1, 8), # Neck → MidHip
    (8, 9), # MidHip → RHip
    (9, 10), # RHip → RKnee
    (10, 11), # RKnee → RAnkle
    (8, 12), # MidHip → LHip
    (12, 13), # LHip → LKnee
    (13, 14), # LKnee → LAnkle
    (0, 15), # Nose → REye
    (0, 16), # Nose → LEye
    (15, 17), # REye → REar
    (16, 18), # LEye → LEar
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # index
    (0, 9), (9, 10), (10, 11), (11, 12), # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]


def _parse_frame(frame: np.ndarray) -> tuple:
    """
    Parses a single (274,) feature vector into body, left hand, right hand joints.

    Returns:
        body:  (25, 2) body joint coordinates
        left:  (21, 2) left hand joint coordinates
        right: (21, 2) right hand joint coordinates
    """
    body  = frame[0:50].reshape(25, 2)
    left  = frame[190:232].reshape(21, 2)
    right = frame[232:274].reshape(21, 2)
    return body, left, right


def _draw_connections(ax, joints: np.ndarray, connections: list, color: str, alpha: float = 0.8):
    """Draws bone lines between connected joints, skipping zero joints."""
    for i, j in connections:
        xi, yi = joints[i]
        xj, yj = joints[j]
        # Skip if either joint is at origin (undetected)
        if abs(xi) < 1e-6 and abs(yi) < 1e-6:
            continue
        if abs(xj) < 1e-6 and abs(yj) < 1e-6:
            continue
        ax.plot([xi, xj], [yi, yj], color=color, linewidth=1.5, alpha=alpha)


def _draw_joints(ax, joints: np.ndarray, color: str, size: float = 20.0):
    """Draws joint dots, skipping zero joints."""
    valid = ~((joints[:, 0] == 0) & (joints[:, 1] == 0))
    ax.scatter(
        joints[valid, 0], joints[valid, 1],
        c=color, s=size, zorder=3
    )


def animate_sequence( sequence: np.ndarray, output_path: str = None, fps: int = 15, figsize: tuple = (4, 6)) -> str:
    """
    Generates an animated GIF of the skeleton from a keypoint sequence. Returns path to the saved GIF file.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        output_path = tmp.name
        tmp.close()

    # Only animate frames with actual signal
    frame_sums   = np.abs(sequence).sum(axis=1)
    valid_frames = np.where(frame_sums > 0)[0]

    if len(valid_frames) == 0:
        # Nothing to animate — return a blank frame
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No keypoints detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.axis("off")
        fig.savefig(output_path, format="gif")
        plt.close(fig)
        return output_path

    frames_to_show = sequence[valid_frames]

    # Compute axis limits from all valid frames for a stable view
    all_body = frames_to_show[:, 0:50].reshape(-1, 2)
    valid_xy = all_body[~((all_body[:, 0] == 0) & (all_body[:, 1] == 0))]

    if len(valid_xy) == 0:
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
    else:
        margin = 0.5
        x_min = valid_xy[:, 0].min() - margin
        x_max = valid_xy[:, 0].max() + margin
        y_min = valid_xy[:, 1].min() - margin
        y_max = valid_xy[:, 1].max() + margin

    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert Y so image coords match (0,0 = top-left)
    ax.axis("off")

    def update(frame_idx):
        ax.cla()
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.axis("off")

        frame = frames_to_show[frame_idx]
        body, left, right = _parse_frame(frame)

        # Body skeleton (white bones, cyan joints)
        _draw_connections(ax, body,  BODY_CONNECTIONS,  color="#ffffff")
        _draw_joints(ax,      body,  color="#00d4ff",   size=25)

        # Left hand (green)
        _draw_connections(ax, left,  HAND_CONNECTIONS,  color="#00ff88", alpha=0.7)
        _draw_joints(ax,      left,  color="#00ff88",   size=12)

        # Right hand (orange)
        _draw_connections(ax, right, HAND_CONNECTIONS,  color="#ff8800", alpha=0.7)
        _draw_joints(ax,      right, color="#ff8800",   size=12)

        # Frame counter
        ax.text(
            0.02, 0.02, f"frame {valid_frames[frame_idx]+1}",
            transform=ax.transAxes, color="#888888",
            fontsize=8, va="bottom"
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames_to_show),
        interval=1000 // fps,
        repeat=True,
    )

    ani.save(output_path, writer="pillow", fps=fps)
    plt.close(fig)

    return output_path


def animate_from_npy(npy_path: str, output_path: str = None, fps: int = 15) -> str:
    """
    Convenience wrapper: loads a .npy file and generates the animation. Returns path to the saved GIF file.
    """
    sequence = np.load(npy_path)
    return animate_sequence(sequence, output_path=output_path, fps=fps)