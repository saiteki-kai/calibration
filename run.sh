#! /bin/bash

dataset="PKU-Alignment/Beavertails"
split="330k_test"
ece_bins=15
plot_bins=20
verbose=False
show_plot=False

hf_name_to_path() {
    local hf_name="$1"
    echo "${hf_name//\//__}"
}

run_eval() {
    local model="$1"
    local taxonomy="$2"
    local descriptions="$3"

    python eval.py --model "${model}" \
        --sample-size 10 \
        --dataset-name "${dataset}" \
        --split "${split}" \
        --taxonomy "${taxonomy}" \
        --descriptions "${descriptions}" \
        --ece-bins "${ece_bins}" \
        --verbose "${verbose}" \
        --output-path "results/$(hf_name_to_path "$model")/${taxonomy}" \
        --show-plot "${show_plot}" \
        --plot-bins "${plot_bins}"
}

# Base model with default taxonomy
run_eval "meta-llama/Llama-Guard-3-1B" "llama-guard-3" "False"

# Base model with Beavertails taxonomy
run_eval "meta-llama/Llama-Guard-3-1B" "beavertails" "False"

# Fine-tuned model with Beavertails taxonomy
run_eval "meta-llama/Llama-Guard-3-8B-Lora" "beavertails" "False"

# # Base model with Beavertails taxonomy and descriptions
# run_eval "meta-llama/Llama-Guard-3-8B" "beavertails" "True"
