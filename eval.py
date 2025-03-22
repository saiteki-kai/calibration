from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.calibration import CalibrationDisplay
import torch

from datasets import Dataset
from matplotlib import pyplot as plt
from sklearn.metrics import auc, brier_score_loss, classification_report, precision_recall_curve
from transformers import set_seed

import calibration as cal

from chat_template import load_chat_template
from utils import generate, load_beavertails, load_model


if __name__ == "__main__":
    set_seed(42)

    calibration_method = "origin"
    model_name = "meta-llama/Llama-Guard-3-1B"
    taxonomy = "llama-guard-3"
    # taxonomy = "beavertails"

    # load model and tokenizer
    model, tokenizer = load_model(model_name)
    tokenizer.chat_template = load_chat_template(taxonomy, descriptions=True)

    model = torch.compile(model, fullgraph=True)
    model.eval()

    # load and prepare dataset
    print("loading dataset...")
    dataset = load_beavertails(tokenizer)
    dataset = dataset.select(range(5000))

    # predict and evaluate
    # compute_predictions(model, dataset)
    tmp_filepath = Path(f"{calibration_method}.jsonl")

    if not tmp_filepath.exists():
        results = dataset.map(partial(generate, model=model, tokenizer=tokenizer, max_length=2048, max_new_tokens=10))
        results = results.remove_columns(["text"])
        results.to_json(tmp_filepath)
    else:
        results = cast(Dataset, Dataset.from_json(str(tmp_filepath)))

    results = cast(dict, results.to_dict())

    model_probs = np.array(results["unsafe_prob"])
    labels = np.array(results["gt_label"])
    pred_labels = np.array(results["pred_label"])

    print(f"ECE: {cal.utils.get_calibration_error(model_probs, labels, debias=False)}")
    print(f"Brier: {brier_score_loss(labels, model_probs)}")

    precision, recall, _ = precision_recall_curve(labels, model_probs)
    print("AUPRC: ", auc(recall, precision))

    print(classification_report(labels, pred_labels))

    fig = plt.figure(figsize=(4, 4))
    # x, y = calibration_curve(labels, model_probs)
    CalibrationDisplay.from_predictions(labels, model_probs, n_bins=20, ax=fig.add_subplot(1, 1, 1))
    # sns.lineplot(x=x, y=y)
    plt.tight_layout()
    plt.show()
