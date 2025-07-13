#!/bin/bash

# This script runs the experiments for the Beavertails dataset.
# It tunes the models, evaluates them, and generates plots and metrics.

hf_name_to_path() {
    local hf_name="$1"
    echo "${hf_name//\//__}"
}

dataset="PKU-Alignment/Beavertails"
dev_split="330k_train" # (use 10% of 330k)
eval_split="330k_test"

taxonomy="beavertails"
descriptions="False"

models=(
    "meta-llama/Llama-Guard-3-8B"
    "saiteki-kai/QA-Llama-Guard-3-8B"
)

ece_bins=15
plot_bins=10
# sample_size=2000 # (for debugging, use a smaller sample size)

evaluation_path="results/evaluation/"
tuning_path="results/tuning/"
comparison_path="results/comparison/"

for model in "${models[@]}"; do
    # 1. Tuning parametrized calibration methods
    python scripts/tuning.py --model "${model}" \
        --dataset-name "${dataset}" \
        --split "${dev_split}" \
        --taxonomy "${taxonomy}" \
        --descriptions "${descriptions}" \
        --ece-bins "${ece_bins}" \
        --output-path "${tuning_path}$(hf_name_to_path "$model")/"

    # Get the best hyperparameters from tuning results
    temperature=$(jq <"${tuning_path}$(hf_name_to_path "$model")/temperature.json" -r '.best')
    gamma=$(jq <"${tuning_path}$(hf_name_to_path "$model")/gamma.json" -r '.best')

    # 2. Evaluate uncalibrated and calibrated models
    python scripts/eval.py --model "${model}" \
        --dataset-name "${dataset}" \
        --split "${eval_split}" \
        --taxonomy "${taxonomy}" \
        --descriptions "${descriptions}" \
        --ece-bins "${ece_bins}" \
        --temperature "${temperature}" \
        --gamma "${gamma}" \
        --output-path "${evaluation_path}$(hf_name_to_path "$model")/"
        # --sample-size ${sample_size} \
done

# 3. Compare models through metrics and plots
python scripts/plots.py --models "${models[@]}" \
    --dataset-name "${dataset}" \
    --split "${eval_split}" \
    --plot-bins "${plot_bins}" \
    --input-path "${evaluation_path}" \
    --output-path "${comparison_path}"
    # --sample-size ${sample_size} \

python scripts/metrics.py \
    --models "${models[@]}" \
    --input-path "${evaluation_path}" \
    --output-path "${comparison_path}/metrics" \
