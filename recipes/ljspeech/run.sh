#!/bin/bash

set -e

echo "===== FastSpeech2 Training Pipeline ====="

CONFIG="configs/config.yaml"

echo ""
echo "Step 1: Downloading LJSpeech dataset..."
bash recipes/ljspeech/prepare.sh

echo ""
echo "Step 2: Preprocessing dataset..."
python fastspeech2/cli/preprocess.py --config $CONFIG --stage 0

echo ""
echo "Step 3: Training FastSpeech2..."
python fastspeech2/cli/train.py --config $CONFIG

echo ""
echo "===== Training Pipeline Completed ====="
