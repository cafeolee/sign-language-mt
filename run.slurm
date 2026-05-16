#!/usr/bin/env bash
#SBATCH --job-name=sign-language-train
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sl-train.log
#SBATCH --error=logs/sl-train.err
#SBATCH --time=96:00:00

mkdir -p /home/your_username/.cache/huggingface
export HF_HOME=/home/your_username/.cache/huggingface
export HF_HUB_CACHE=/home/your_username/.cache/huggingface
export HF_USERNAME=your_username

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /your/path
source /your/path/.venv/bin/activate

export PYTHONPATH=/your/path:$PYTHONPATH

python train.py