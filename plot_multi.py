from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path("/content/verification-drop-llm/data/multi_model_results.json")
OUTPUT_PATH = Path("/content/verification-drop-llm/paper/multi_model_comparison.png")

MODEL_STYLES = {
    "deepseek-r1:1.5b": {"color": "#1f77b4", "marker": "o"},
    "deepseek-r1:8b": {"color": "#d62728", "marker": "s"},
    "qwen2.5-math:7b": {"color": "#2ca02c", "marker": "^"},
}


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found at {RESULTS_PATH}. Run `python src/evaluate_multi.py` first."
        )
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def plot(results: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 6), dpi=300)
    condition_order = ["Control", "Low", "Medium", "High"]

    for model, condition_results in results["results"].items():
        x_values = np.array(
            [condition_results[condition]["noise_token_count"] for condition in condition_order],
            dtype=float,
        )
        y_values = np.array(
            [
                condition_results[condition]["mean_verification_density_score"]
                for condition in condition_order
            ],
            dtype=float,
        )
        style = MODEL_STYLES.get(model, {"color": None, "marker": "o"})
        ax.plot(
            x_values,
            y_values,
            label=model,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=7,
        )

    ax.set_title("Verification Drop Tracking Across Contextual Noise Conditions", pad=14)
    ax.set_xlabel("Contextual Noise Size (Tokens)")
    ax.set_ylabel(r"Mean Verification Density Score ($D_v$)")
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(title="Model", frameon=True, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = load_results()
    plot(results)
    print(f"Wrote plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
