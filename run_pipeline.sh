#!/bin/bash

# ====================================================================
# Master Pipeline Script for the Protein Localization Predictor v2.0
# ====================================================================

# Stop the script if any command fails
set -e

# --- CONFIGURATION ---
# Default number of samples to download if none is provided.
# Use -1 for all available samples.
DEFAULT_SAMPLES=2000 # Aumentado para aprovechar Deep Learning
MIN_SAMPLES_PER_CLASS=50

# Check if a command-line argument was provided for the number of samples
SAMPLES=${1:-$DEFAULT_SAMPLES}

# --- PIPELINE STEPS ---

echo "🚀 STARTING MODERN PIPELINE (ESM-2 + BioPhysics) WITH $SAMPLES SAMPLES 🚀"

# Step 1: Download Data (Multi-Species)
echo "-------------------------------------"
echo "STEP 1: Downloading multi-species data from UniProt..."
echo "-------------------------------------"
python scripts/download_data.py --max_results $SAMPLES

# Data Bridge: Ensure filenames match between steps
# El nuevo downloader guarda como 'uniprot_data_multispecies.csv'
# El data_processing espera 'uniprot_data.csv'
if [ -f "data/raw/uniprot_data_multispecies.csv" ]; then
    echo "🔄 Bridging data files..."
    mv data/raw/uniprot_data_multispecies.csv data/raw/uniprot_data.csv
fi

# Step 2: Process Data
echo "-------------------------------------"
echo "STEP 2: Processing, cleaning and hierarchical grouping..."
echo "-------------------------------------"
python src/dataset.py --min_samples $MIN_SAMPLES_PER_CLASS

# Step 3: End-to-End Training (Replaces old Embedding + XGBoost steps)
echo "-------------------------------------"
echo "STEP 3: Training Hybrid Model (Fine-Tuning ESM-2 + Bio-Features)..."
echo "NOTE: This utilizes GPU if available. It effectively replaces embedding generation."
echo "-------------------------------------"
python src/train_hybrid.py

# Optional: Explainability Report
if [ -f "src/explainability.py" ]; then
    echo "-------------------------------------"
    echo "STEP 4: Generating Explainability Reports (SHAP)..."
    echo "-------------------------------------"
    python src/explainability.py # Descomenta si quieres generar gráficos automáticamente
    # echo "Skipping report generation (run 'python src/explainability.py' manually)."
fi

echo "✅ PIPELINE COMPLETED SUCCESSFULLY! ✅"
echo "The new hybrid model is ready in 'models/esm2_hybrid_finetuned/'"

# Build and start Docker container
echo "-------------------------------------"
echo "STEP 5: Building and starting API container..."
echo "-------------------------------------"
if ! docker images | grep -q "protein-location-predictor"; then
    echo "🔨 Building Docker image..."
    docker-compose build
fi
echo "🚀 Starting API container..."
docker-compose up -d
echo "✅ API is now running! Access it at http://localhost:8000"