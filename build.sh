#!/bin/bash

cp .env-example .env

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

DATA_DIR="${DATA_DIR:-./data}"
DATASET_ZIP="${DATASET_ZIP:-$DATA_DIR/dataset.zip}"
DATASET_URL="${DATASET_URL}"

if [ -z "$DATASET_URL" ]; then
    echo "Error: DATASET_URL environment variable must be set"
    echo "Example: DATASET_URL='https://example.com/dataset.zip' ./script.sh"
    exit 1
fi

mkdir -p "$DATA_DIR"

if [ ! -f "$DATASET_ZIP" ]; then
    echo "Downloading dataset from $DATASET_URL..."
    if ! curl -L -o "$DATASET_ZIP" "$DATASET_URL"; then
        echo "Error: Failed to download dataset"
        rm -f "$DATASET_ZIP"
        exit 1
    fi
else
    echo "Dataset already downloaded at $DATASET_ZIP"
fi

if [ -f "$DATASET_ZIP" ] && [ -z "$(find "$DATA_DIR" -maxdepth 1 -type f ! -name "$(basename "$DATASET_ZIP")")" ]; then
    echo "Extracting dataset..."
    if ! unzip -q "$DATASET_ZIP" -d "$DATA_DIR"; then
        echo "Error: Failed to extract dataset"
        exit 1
    fi
    echo "Extraction complete"
else
    echo "Data directory already contains files, skipping extraction"
fi

# Copy this to build manualy
docker compose build