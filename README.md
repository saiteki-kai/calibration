# Llama Guard 3 Calibration Project

A comprehensive framework for assessing calibration quality in Llama Guard 3 models on the BeaverTails safety taxonomy.

This project evaluates both base Llama Guard 3 with in-context learning and fine-tuned Llama Guard 3 variants trained on the BeaverTails dataset.

The framework implements multiple calibration techniques including temperature scaling, contextual calibration, and batch calibration to analyze the reliability of safety predictions in content moderation systems.

The framework focuses on binary safety classification, analyzing how well models distinguish between safe and unsafe content. All evaluation metrics and visualizations are reported with respect to the positive (unsafe) class.

## Models

The project evaluates the following models:

- [meta-llama/Llama-Guard-3-8B](https://huggingface.com/meta-llama/Llama-Guard-3-8B): Base model with in-context learning
- [saiteki-kai/QA-Llama-Guard-3-8B](https://huggingface.com/saiteki-kai/QA-Llama-Guard-3-8B): Fine-tuned variant trained on the BeaverTails dataset

## Calibration Methods

The project implements three calibration methods:

1. **Temperature Scaling**: Scales logits by a learned temperature parameter
2. **Batch Calibration**: Adjusts predictions using distribution-level statistics
3. **Context-Free Calibration**: Calibrates based on token-level probabilities

## Metrics

The framework evaluates both calibration and classification performance:

**Calibration Metrics:**

- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Negative Log-Likelihood (NLL)

**Classification Metrics:**

- Accuracy
- F1 Score
- Precision
- Recall
- Area Under Precision-Recall Curve (AUPRC)

## Project Structure

```text
.
├── src/
│   ├── core/
│   │   ├── calibrators/        # Calibration implementations
│   │   ├── classifiers/        # Model wrappers
│   │   ├── types.py            # Data structures
│   │   └── utils.py            # Utility functions
│   └── evaluation/
│       ├── metrics.py          # Evaluation metrics
│       └── visualization/      # Plotting utilities
├── scripts/
│   ├── tuning.py               # Parameter tuning
│   ├── eval.py                 # Model evaluation
│   ├── plots.py                # Plots generation
│   └── metrics.py              # Metrics table generation
├── results/                    # Experiment outputs and plots
├── tests/                      # Unit tests
├── requirements.txt
├── pyproject.toml
└── run_experiments.sh
```

## Requirements

- Python ≥ 3.12
- PyTorch
- Transformers
- Datasets
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas

## Installation

Clone the repository:

```bash
git clone https://github.com/saiteki-kai/llama_guard_calibration
cd llama_guard_calibration
```

Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

Run the complete experiment pipeline (parameter tuning, evaluation, and plots/tables generation):

```bash
./run_experiments.sh
```

### Individual Scripts

#### 1. Parameter Tuning

Tune calibration parameters on validation data:

```bash
python scripts/tuning.py \
    --model meta-llama/Llama-Guard-3-8B \
    --dataset-name PKU-Alignment/Beavertails \
    --split 330k_train \
    --taxonomy beavertails \
    --descriptions False \
    --ece-bins 15 \
    --seed 42 \
    --test-size 0.1 \
    --output-path results/tuning/
```

#### 2. Model Evaluation

Evaluate calibrated and uncalibrated models:

```bash
python scripts/eval.py \
    --model meta-llama/Llama-Guard-3-8B \
    --dataset-name PKU-Alignment/Beavertails \
    --split 330k_test \
    --taxonomy beavertails \
    --descriptions False \
    --ece-bins 15 \
    --temperature 5.0 \
    --gamma 1.0 \
    --seed 42 \
    --max-new-tokens 10 \
    --output-path results/evaluation/
```

#### 3. Generate Plots

Create reliability diagrams and calibration curves:

```bash
python scripts/plots.py \
    --models meta-llama/Llama-Guard-3-8B saiteki-kai/QA-Llama-Guard-3-8B \
    --dataset-name PKU-Alignment/Beavertails \
    --split 330k_test \
    --input-path results/evaluation/ \
    --output-path results/comparison/plots/ \
    --plot-bins 10
```

#### 4. Generate Metrics Table

Create LaTeX metrics table:

```bash
python scripts/metrics.py \
    --models meta-llama/Llama-Guard-3-8B saiteki-kai/QA-Llama-Guard-3-8B \
    --input-path results/evaluation/ \
    --output-path results/comparison/metrics/
```
