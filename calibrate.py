from typing import cast
from datasets import Dataset
from matplotlib import pyplot as plt
import torch
from transformers import set_seed
import numpy as np
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE

from chat_template import load_chat_template
from utils import generate, load_model, prepare_input


def calibrate_py(p_y, p_cf, mode="diagonal"):
    num_classes = p_y.shape[0]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
            cal_py = np.matmul(W, np.expand_dims(p_y, axis=-1)) + b
        elif mode == "identity":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(np.log(p_cf), axis=-1)
            cal_py = np.matmul(W, np.expand_dims(np.log(p_y + 10e-6), axis=-1)) + b
            cal_py = np.exp(cal_py)

    cal_py = cal_py / np.sum(cal_py)

    return cal_py


def compute_contextual_free_prob(tokenizer, token: str = "N/A"):
    data = {"prompt": token, "response": token, "gt_label": 0, "is_safe": True}
    x = prepare_input(data, tokenizer)
    return generate(x, model, tokenizer)
    
    # print(x["text"])


if __name__ == "__main__":
    set_seed(42)  # may not be necessary

    model_name = "meta-llama/Llama-Guard-3-1B"
    taxonomy = "llama-guard-3"
    # taxonomy = "beavertails"

    # load model and tokenizer
    model, tokenizer = load_model(model_name)
    tokenizer.chat_template = load_chat_template(taxonomy, descriptions=True)

    model = torch.compile(model, fullgraph=True)
    model.eval()

    print(compute_contextual_free_prob(tokenizer, token=" "))
    exit()

    #######################ààààà

    dataset = Dataset.from_json("origin.jsonl")
    dataset = cast(Dataset, dataset)

    model_probs = np.array(dataset["unsafe_prob"])
    labels = np.array(dataset["gt_label"])
    pred_labels = np.array(dataset["pred_label"])

    # batch calibration (maybe i should use a calibration set to estimate this)
    batch_prob = np.mean(np.stack([model_probs, 1 - model_probs], axis=1), axis=0)
    print(batch_prob)

    probs = []
    pred_labels_new = []
    for i in range(model_probs.shape[0]):
        prob = np.asarray([model_probs[i], 1 - model_probs[i]])
        cal_prob = calibrate_py(prob, batch_prob, mode="diagonal")
        probs.append(cal_prob[0][0])
        pred_lbl = int(np.argmin(cal_prob.reshape(-1)))
        pred_labels_new.append(pred_lbl)

    c_model_probs = np.array(probs)
    c_pred_labels = np.array(pred_labels_new)

    print(ECE(bins=15).measure(model_probs, labels))
    print(ECE(bins=15).measure(c_model_probs, labels))
    import calibration

    print(f"ECE: {calibration.utils.get_calibration_error(model_probs, labels, debias=False)}")
    print(f"ECE: {calibration.utils.get_calibration_error(c_model_probs, labels, debias=False)}")

    diagram = ReliabilityDiagram(bins=15)
    diagram.plot(model_probs, labels)
    diagram.plot(c_model_probs, labels)
    plt.show()
