# Makefile for Protein Localization Predictor v2.0
# Complete pipeline automation

.PHONY: help setup install download preprocess features train evaluate export deploy clean

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Configuration
PYTHON := python3
PIP := pip3
CONFIG_DIR := config
DATA_DIR := data
MODELS_DIR := models
SCRIPTS_DIR := scripts

# Default target
help:
	@echo "$(GREEN)Protein Localization Predictor v2.0$(NC)"
	@echo ""
	@echo "Available targets:"
	@echo "  $(YELLOW)setup$(NC)          - Create directory structure"
	@echo "  $(YELLOW)install$(NC)        - Install all dependencies"
	@echo "  $(YELLOW)download$(NC)       - Download protein data from all sources"
	@echo "  $(YELLOW)preprocess$(NC)     - Clean and split data with GraphPart"
	@echo "  $(YELLOW)features$(NC)       - Extract multimodal features"
	@echo "  $(YELLOW)train$(NC)          - Train hierarchical model"
	@echo "  $(YELLOW)evaluate$(NC)       - Evaluate on test set"
	@echo "  $(YELLOW)export$(NC)         - Export to ONNX format"
	@echo "  $(YELLOW)deploy$(NC)         - Build and deploy Docker container"
	@echo "  $(YELLOW)full-pipeline$(NC)  - Run complete pipeline end-to-end"
	@echo "  $(YELLOW)clean$(NC)          - Remove generated files"
	@echo ""
	@echo "Quick start: $(GREEN)make full-pipeline$(NC)"

# Setup project structure
setup:
	@echo "$(GREEN)Creating project structure...$(NC)"
	mkdir -p $(DATA_DIR)/{raw,interim,processed,external,partitions}
	mkdir -p $(MODELS_DIR)/{checkpoints,final,pretrained}
	mkdir -p reports/{figures,metrics,experiment_logs}
	mkdir -p logs
	mkdir -p notebooks
	mkdir -p tests
	@echo "$(GREEN)✓ Project structure created$(NC)"

# Install dependencies
install: setup
	@echo "$(GREEN)Installing Python dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Installing external tools...$(NC)"
	@echo "Please install manually:"
	@echo "  - MMseqs2: conda install -c bioconda mmseqs2"
	@echo "  - GraphPart: pip install graph-part"
	@echo "  - SignalP 6.0: https://services.healthtech.dtu.dk/services/SignalP-6.0/"
	@echo "  - TMHMM 2.0: https://services.healthtech.dtu.dk/services/TMHMM-2.0/"
	@echo "$(GREEN)✓ Python dependencies installed$(NC)"

# Download data from multiple sources
download:
	@echo "$(GREEN)Downloading protein data...$(NC)"
	@echo "This may take 30-60 minutes depending on sample size"
	$(PYTHON) $(SCRIPTS_DIR)/01_download_data.py --config $(CONFIG_DIR)/data_sources.yaml
	@echo "$(GREEN)✓ Data download complete$(NC)"

# Preprocess and split data
preprocess:
	@echo "$(GREEN)Preprocessing and splitting data...$(NC)"
	@echo "Using GraphPart for homology-aware splitting..."
	$(PYTHON) $(SCRIPTS_DIR)/02_preprocess_data.py \
		--input $(DATA_DIR)/raw/uniprot_data_multispecies.csv \
		--output-dir $(DATA_DIR)/processed \
		--config $(CONFIG_DIR)/base_config.yaml \
		--method graphpart
	@echo "$(GREEN)✓ Preprocessing complete$(NC)"

# Extract features from all modalities
features:
	@echo "$(GREEN)Extracting multimodal features...$(NC)"
	@echo "This will take significant time (2-4 hours with ESM-2)"
	# Train set
	$(PYTHON) $(SCRIPTS_DIR)/03_generate_features.py \
		--input $(DATA_DIR)/processed/train_data.csv \
		--output $(DATA_DIR)/processed/train_features.csv \
		--config $(CONFIG_DIR)/training_config.yaml
	# Validation set
	$(PYTHON) $(SCRIPTS_DIR)/03_generate_features.py \
		--input $(DATA_DIR)/processed/val_data.csv \
		--output $(DATA_DIR)/processed/val_features.csv \
		--config $(CONFIG_DIR)/training_config.yaml
	# Test set
	$(PYTHON) $(SCRIPTS_DIR)/03_generate_features.py \
		--input $(DATA_DIR)/processed/test_data.csv \
		--output $(DATA_DIR)/processed/test_features.csv \
		--config $(CONFIG_DIR)/training_config.yaml
	@echo "$(GREEN)✓ Feature extraction complete$(NC)"

# Train model
train:
	@echo "$(GREEN)Training hierarchical model...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/04_train_model.py \
		--config $(CONFIG_DIR)/base_config.yaml
	@echo "$(GREEN)✓ Training complete$(NC)"

# Evaluate model
evaluate:
	@echo "$(GREEN)Evaluating model on test set...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/05_evaluate_model.py \
		--checkpoint $(MODELS_DIR)/checkpoints/best.ckpt \
		--test-data $(DATA_DIR)/processed/test_features.csv \
		--output-dir reports/figures
	@echo "$(GREEN)✓ Evaluation complete$(NC)"

# Export to ONNX
export:
	@echo "$(GREEN)Exporting model to ONNX...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/06_export_onnx.py \
		--checkpoint $(MODELS_DIR)/final/final_model.ckpt \
		--output $(MODELS_DIR)/final/model.onnx \
		--quantize int8
	@echo "$(GREEN)✓ Model exported$(NC)"

# Build and deploy Docker container
deploy:
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker-compose build
	@echo "$(GREEN)Starting containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Deployment complete$(NC)"
	@echo "API available at: http://localhost:8000"
	@echo "Health check: curl http://localhost:8000/health"

# Run complete pipeline
full-pipeline: setup download preprocess features train evaluate export
	@echo "$(GREEN)════════════════════════════════════════$(NC)"
	@echo "$(GREEN)✓ Full pipeline completed successfully!$(NC)"
	@echo "$(GREEN)════════════════════════════════════════$(NC)"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Review results in reports/figures/"
	@echo "  2. Deploy with: make deploy"
	@echo "  3. Test API with: scripts/test_api.sh"

# Testing
test:
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html
	@echo "$(GREEN)✓ Tests complete$(NC)"

# Lint code
lint:
	@echo "$(GREEN)Linting code...$(NC)"
	black src/ scripts/ --check
	flake8 src/ scripts/ --max-line-length=100
	mypy src/ --ignore-missing-imports

# Format code
format:
	@echo "$(GREEN)Formatting code...$(NC)"
	black src/ scripts/
	isort src/ scripts/

# Clean generated files
clean:
	@echo "$(RED)Cleaning generated files...$(NC)"
	rm -rf $(DATA_DIR)/interim/*
	rm -rf $(DATA_DIR)/processed/*
	rm -rf $(DATA_DIR)/partitions/*
	rm -rf $(MODELS_DIR)/checkpoints/*
	rm -rf logs/*
	rm -rf reports/figures/*
	rm -rf reports/metrics/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Deep clean (including raw data and models)
deep-clean: clean
	@echo "$(RED)Deep cleaning (including raw data and models)...$(NC)"
	rm -rf $(DATA_DIR)/raw/*
	rm -rf $(MODELS_DIR)/final/*
	rm -rf $(MODELS_DIR)/pretrained/*
	@echo "$(GREEN)✓ Deep cleanup complete$(NC)"

# Quick test run (1% of data for debugging)
quick-test:
	@echo "$(YELLOW)Running quick test with 1% of data...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/01_download_data.py --max-results 100
	$(PYTHON) $(SCRIPTS_DIR)/02_preprocess_data.py \
		--input $(DATA_DIR)/raw/uniprot_data_multispecies.csv \
		--output-dir $(DATA_DIR)/processed \
		--method random
	$(PYTHON) $(SCRIPTS_DIR)/04_train_model.py \
		--config $(CONFIG_DIR)/base_config.yaml \
		--fast-dev-run
	@echo "$(GREEN)✓ Quick test complete$(NC)"

# Download pretrained models (if available)
download-pretrained:
	@echo "$(GREEN)Downloading pretrained models...$(NC)"
	# Add script to download from Hugging Face or cloud storage
	@echo "$(YELLOW)Not implemented yet$(NC)"

# Generate documentation
docs:
	@echo "$(GREEN)Generating documentation...$(NC)"
	cd docs && mkdocs build
	@echo "$(GREEN)✓ Documentation generated$(NC)"
	@echo "Open: docs/site/index.html"

# Monitor training with TensorBoard
tensorboard:
	@echo "$(GREEN)Starting TensorBoard...$(NC)"
	tensorboard --logdir=reports/experiment_logs --port=6006
	@echo "Open: http://localhost:6006"

# Profile model performance
profile:
	@echo "$(GREEN)Profiling model performance...$(NC)"
	$(PYTHON) -m torch.utils.bottleneck $(SCRIPTS_DIR)/04_train_model.py \
		--fast-dev-run
	@echo "$(GREEN)✓ Profiling complete$(NC)"

# Check dependencies
check-deps:
	@echo "$(GREEN)Checking dependencies...$(NC)"
	@command -v mmseqs >/dev/null 2>&1 || echo "$(RED)✗ MMseqs2 not found$(NC)"
	@command -v graph-part >/dev/null 2>&1 || echo "$(RED)✗ GraphPart not found$(NC)"
	@command -v signalp6 >/dev/null 2>&1 || echo "$(RED)✗ SignalP 6.0 not found$(NC)"
	@command -v tmhmm >/dev/null 2>&1 || echo "$(RED)✗ TMHMM not found$(NC)"
	@$(PYTHON) -c "import torch; print('✓ PyTorch:', torch.__version__)"
	@$(PYTHON) -c "import transformers; print('✓ Transformers:', transformers.__version__)"
	@echo "$(GREEN)✓ Dependency check complete$(NC)"
