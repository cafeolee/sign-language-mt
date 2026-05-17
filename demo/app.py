"""
app.py
------
Gradio demo for the Sign Language Translation model.

The user records a short video (or uploads one), the app extracts
body and hand keypoints with MediaPipe, runs the translation model,
and displays the skeleton animation alongside the predicted text.

Run locally:
    python demo/app.py

Run with public share URL (for poster session):
    python demo/app.py --share
"""

import sys
import argparse
from pathlib import Path

# Make src/ importable from demo/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import gradio as gr

from demo.inference import load_model, predict
from demo.mediapipe_adapter import video_to_keypoints
from demo.skeleton import animate_sequence

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "config.yaml"
CHECKPOINT   = PROJECT_ROOT / "checkpoints" / "best_model.pt"

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
print("Loading model...")
model, tokenizer, config, device = load_model(
    config_path=str(CONFIG_PATH),
    checkpoint_path=str(CHECKPOINT),
)
print("Model ready.")

# ---------------------------------------------------------------------------
# Core translation function
# ---------------------------------------------------------------------------
def translate_video(video_path: str):
    """
    Receives a video file path from Gradio, extracts keypoints with MediaPipe,
    animates the skeleton, runs the model, and returns (gif_path, prediction).
    """
    if video_path is None:
        return None, "⚠️ Please record or upload a video first."

    max_kp_frames = config["data"].get("max_frames", 150)
    sequence      = video_to_keypoints(video_path, max_frames=max_kp_frames)

    if sequence is None:
        return None, "⚠️ Could not read the video. Please try again."

    valid_frames = (np.abs(sequence).sum(axis=1) > 0).sum()
    if valid_frames == 0:
        return None, "⚠️ No person detected. Make sure you are fully visible and well-lit."

    gif_path   = animate_sequence(sequence, fps=15)
    prediction = predict(sequence, model, tokenizer, config, device)

    return gif_path, prediction


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="Sign Language Translation", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🤟 Sign Language Translation
        Record a short ASL video (3–5 seconds). The model extracts your body
        and hand keypoints and translates your signs into English text.

        **Tips:**
        - Make sure your full upper body is visible
        - Use good lighting
        - Sign clearly and at a moderate pace
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="Record or upload a video",
                sources=["webcam", "upload"],
                format="mp4",
            )
            translate_btn = gr.Button("🤟 Translate", variant="primary", size="lg")

        with gr.Column(scale=1):
            skeleton_out = gr.Image(
                label="What the model sees",
                type="filepath",
            )
            prediction_out = gr.Textbox(
                label="Translation",
                lines=3,
                interactive=False,
                placeholder="Translation will appear here...",
            )

    translate_btn.click(
        fn=translate_video,
        inputs=[video_input],
        outputs=[skeleton_out, prediction_out],
    )

    gr.Markdown(
        """
        ---
        *This model was trained on the [How2Sign](https://how2sign.github.io/) dataset
        using pose keypoints extracted with OpenPose.
        Keypoints are extracted here in real time using MediaPipe.*
        """
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Generate a public share URL")
    parser.add_argument("--port",  type=int, default=7860)
    args = parser.parse_args()

    demo.launch(
        share=args.share,
        server_port=args.port,
        show_error=True,
    )