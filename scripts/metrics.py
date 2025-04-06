import argparse
import json

from pathlib import Path


def generate_latex_metrics_table(
    models: list[str],
    cal_metrics: list[str],
    cls_metrics: list[str],
    short_names: dict[str, str],
    input_path: Path,
    output_path: Path,
    float_fmt: str = "{:.3f}",
) -> None:
    metric_names = [*cal_metrics, *cls_metrics]
    cap_metric_names = [m.upper() if len(m) < 6 else m.capitalize() for m in metric_names]

    filepath = output_path / "metrics.tex"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding="utf-8") as tex_file:
        tex_file.write("\\documentclass{standalone}\n")
        tex_file.write("\\usepackage{booktabs}\n")
        tex_file.write("\\begin{document}\n")
        tex_file.write("\\begin{tabular}{l|" + "c" * len(cal_metrics) + "|" + "c" * len(cls_metrics) + "}\n")
        tex_file.write("\\toprule\n")
        tex_file.write("Model & " + " & ".join(cap_metric_names) + " \\\\\n")
        tex_file.write("\\midrule\n")

        for i, model in enumerate(models):
            # Load metrics for the current model
            metrics = load_metrics(input_path, model.replace("/", "__"))

            for method, method_metrics in metrics.items():
                method_name = model.split("/")[1] if short_names[method] == "" else short_names[method]
                tex_file.write(method_name + " & ")

                values = [float_fmt.format(method_metrics[metric]) for metric in metric_names]
                tex_file.write(" & ".join(values) + " \\\\\n")

            # Add midrule (except for the last model)
            if i < len(models) - 1:
                tex_file.write("\\midrule\n")

        tex_file.write("\\bottomrule\n")
        tex_file.write("\\end{tabular}\n")
        tex_file.write("\\end{document}\n")


def load_metrics(input_path: Path, model: str) -> dict[str, dict[str, float]]:
    filepath = input_path / model / "metrics.json"

    with filepath.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def main(args: argparse.Namespace) -> None:
    args.input_path = Path(args.input_path)
    args.output_path = Path(args.output_path)

    cls_metrics = ["f1", "precision", "recall", "accuracy", "auprc"]
    cal_metrics = ["ece", "mce"]  # "brier"

    short_names = {
        "uncalibrated": "",
        "batch": "+BC",
        "temperature": "+TS",
        "context-free": "+CC",
    }

    generate_latex_metrics_table(
        args.models,
        cal_metrics,
        cls_metrics,
        short_names,
        input_path=args.input_path,
        output_path=args.output_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LaTeX summary table")
    parser.add_argument(
        "--input-path",
        type=str,
        default="results/evaluation",
        help="Path to the input directory containing model metrics",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/comparison/metrics",
        help="Path to the output directory for the LaTeX file",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["meta-llama/Llama-Guard-3-8B"],
        help="List of model names to include in the table",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
