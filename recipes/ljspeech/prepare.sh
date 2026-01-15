#!/bin/bash

DATASET_DIR="data/raw/ljspeech"

mkdir -p $DATASET_DIR

echo "Downloading LJSpeech dataset..."
wget -P $DATASET_DIR https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

echo "Extracting dataset..."
tar -xjf $DATASET_DIR/LJSpeech-1.1.tar.bz2 -C $DATASET_DIR

mv $DATASET_DIR/LJSpeech-1.1/* $DATASET_DIR/
rm -rf $DATASET_DIR/LJSpeech-1.1
rm $DATASET_DIR/LJSpeech-1.1.tar.bz2

echo "LJSpeech dataset downloaded and extracted to $DATASET_DIR"
