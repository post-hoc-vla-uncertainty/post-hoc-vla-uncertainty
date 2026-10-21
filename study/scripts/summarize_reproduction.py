"""Summarize observed SAVE results without declaring reproduction success."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER_FINAL_PERCENT = {"random": (54.6, 0.9), "vfd": (67.1, 3.2), "gu": (64.0, 2.6)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    rows, final_rows = [], []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in args.methods:
        by_round = {}
        for seed in args.seeds:
            path = args.root / method / f"seed_{seed}" / "run_evaluation.json"
            if not path.is_file():
                print(f"Missing evaluation: {path}")
                continue
            for record in json.loads(path.read_text())["rounds"]:
                by_round.setdefault(record["round_index"], []).append(
                    (seed, 100 * record["mean_member_macro_success_rate"])
                )
        method_rows = []
        for round_index, values in sorted(by_round.items()):
            rates = [rate for _, rate in values]
            row = {"method": method, "round_index": round_index,
                   "query_slots": 5 * (round_index + 1),
                   "num_seeds": len(values), "seeds": ";".join(str(seed) for seed, _ in values),
                   "success_percent_mean": float(np.mean(rates)),
                   "success_percent_std": float(np.std(rates))}
            rows.append(row)
            method_rows.append(row)
            if round_index == 14:
                paper = PAPER_FINAL_PERCENT.get(method)
                final_rows.append({**row, "paper_success_percent_mean": paper[0] if paper else None,
                                   "paper_success_percent_std": paper[1] if paper else None,
                                   "all_three_seeds": {seed for seed, _ in values} == {1000, 1001, 1002}})
        if method_rows and method != "calibration_random":
            x = [r["query_slots"] for r in method_rows]
            mean = np.array([r["success_percent_mean"] for r in method_rows])
            std = np.array([r["success_percent_std"] for r in method_rows])
            line, = ax.plot(x, mean, label=method)
            ax.fill_between(x, mean-std, mean+std, alpha=0.15, color=line.get_color())
    if not rows:
        raise RuntimeError("No evaluation results found; run training/evaluation first.")
    output = args.root / "summary"
    output.mkdir(exist_ok=True)
    with (output / "learning_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "final_success.json").write_text(json.dumps(final_rows, indent=2) + "\n")
    if ax.lines:
        ax.set(xlabel="Demonstration query slots (upstream protocol)", ylabel="Macro success (%)",
               ylim=(0, 100), title="SAVE reproduction: mean ± population SD across available seeds")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "learning_curves.png", dpi=160)
        fig.savefig(output / "learning_curves.pdf")
    plt.close(fig)
    for row in final_rows:
        print(f"{row['method']}: {row['success_percent_mean']:.2f} ± {row['success_percent_std']:.2f}% "
              f"({row['num_seeds']} seeds), paper: {PAPER_FINAL_PERCENT.get(row['method'], 'separate calibration trajectory')}")
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()
