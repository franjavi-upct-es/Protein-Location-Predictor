#!/bin/bash

# ====================================================================
# Master Pipeline Script for the Protein Localization Predictor
# ====================================================================

# Stop the script if any command fails
set -e

# --- CONFIGURATION ---
# Default number of samples to download if none is provided.
# Use -1 for all available samples.
DEFAULT_SAMPLES=500
MIN_SAMPLES_PER_CLASS=50

# Check if a command-line argument was provided for the number of samples
SAMPLES=${1:-$DEFAULT_SAMPLES}

# --- PIPELINE STEPS ---

echo "🚀 STARTING PIPELINE WITH $SAMPLES SAMPLES 🚀"
echo "📊 Using Combined Features: Embeddings (ESM-2) + K-mers"

# Step 1: Download Data
echo "-------------------------------------"
echo "STEP 1: Downloading data from UniProt..."
echo "-------------------------------------"
python scripts/download_data.py --max_results $SAMPLES

# Step 2: Process Data
echo "-------------------------------------"
echo "STEP 2: Processing and cleaning data..."
echo "-------------------------------------"
python src/data_processing.py --min_samples $MIN_SAMPLES_PER_CLASS

# Step 3: Generate Embeddings
echo "-------------------------------------"
echo "STEP 3: Generating protein embeddings and k-mers (this may take a while)..."
echo "-------------------------------------"
python src/embedding_generator.py

# Step 4: Train Model
echo "-------------------------------------"
echo "STEP 4: Training the final model with combined features..."
echo "-------------------------------------"
python src/train.py

echo "✅ PIPELINE COMPLETED SUCCESSFULLY! ✅"
echo "Model and results are ready."