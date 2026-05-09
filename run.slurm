#!/usr/bin/env bash
#SBATCH --job-name=sign-language-train
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120GB
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sl-rain.log
#SBATCH --error=logs/sl-train.err
#SBATCH --time=96:00:00

mkdir -p /home/imartinez374/.cache/huggingface
export HF_HOME=/home/imartinez374/.cache/huggingface
export HF_HUB_CACHE=/home/imartinez374/.cache/huggingface
export HF_USERNAME=cafeolee

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/imartinez374/sign-language-mt
source /home/imartinez374/sign-language-mt/venv/bin/activate

export PYTHONPATH=/home/imartinez374/sign-language:$PYTHONPATH

python train.py