import json

from pathlib import Path


def generate_metrics_table(
    models: list[str],
    taxonomy: str,
    cal_metrics: list[str],
    cls_metrics: list[str],
    short_names: dict[str, str],
    float_fmt: str = "{:.3f}",
) -> None:
    metric_names = [*cal_metrics, *cls_metrics]

    with Path("results/comparison/metrics/metrics.tex").open("w", encoding="utf-8") as tex_file:
        tex_file.write("\\documentclass{standalone}\n")
        tex_file.write("\\usepackage{booktabs}\n")
        tex_file.write("\\begin{document}\n")
        tex_file.write("\\begin{tabular}{l|" + "c" * len(cal_metrics) + "|" + "c" * len(cls_metrics) + "}\n")
        tex_file.write("\\toprule\n")
        tex_file.write("Model & " + " & ".join(metric_names) + " \\\\\n")
        tex_file.write("\\midrule\n")

        for i, model in enumerate(models):
            # Load metrics for the current model
            with Path(f"results/{model}/{taxonomy}/metrics.json").open("r", encoding="utf-8") as json_file:
                metrics = json.load(json_file)

            for method, method_metrics in metrics.items():
                method_name = model.split("__")[1] if short_names[method] == "" else short_names[method]
                tex_file.write(method_name + " & ")

                values = [float_fmt.format(method_metrics[metric]) for metric in metric_names]
                tex_file.write(" & ".join(values) + " \\\\\n")

            # Add midrule (except for the last model)
            if i < len(models) - 1:
                tex_file.write("\\midrule\n")

        tex_file.write("\\bottomrule\n")
        tex_file.write("\\end{tabular}\n")
        tex_file.write("\\end{document}\n")


def main() -> None:
    models = ["meta-llama__Llama-Guard-3-1B"]
    taxonomy = "beavertails"

    cls_metrics = ["f1", "precision", "recall", "accuracy", "auprc"]
    cal_metrics = ["ece", "mce"]  # "brier"

    short_names = {
        "uncalibrated": "",
        "batch": "+BC",
        "temperature": "+TS",
        "context-free": "+CC",
    }

    generate_metrics_table(models, taxonomy, cal_metrics, cls_metrics, short_names)


if __name__ == "__main__":
    main()
