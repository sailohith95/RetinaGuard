"""
RetinaGuard — Dataset Inspector (Phase 2)
==========================================
Standalone script to inspect the bumbledeep/aptos HuggingFace dataset
before training. Prints structure, class distribution, and sample info.

Usage:
    python python/dataset/dataset_inspector.py
"""

import json
import logging
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
PYTHON_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retinaguard.inspect")


def main():
    config_path = PYTHON_DIR / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(PROJECT_ROOT)

    from dataset.aptos_loader import APTOSLoader
    loader = APTOSLoader(cfg)

    print("\nPhase 2: Loading and inspecting APTOS dataset…")
    print("(First run downloads dataset; subsequent runs use cache)\n")

    report = loader.inspect()

    # Save
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "dataset_inspection.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nInspection report saved to: {out_path}")

    # Class distribution bar chart
    _plot_class_distribution(report, results_dir)


def _plot_class_distribution(report: dict, results_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
        COLORS = ["#2d7a2d", "#c47d00", "#c47d00", "#b53000", "#8b0000"]

        fig, axes = plt.subplots(1, len(report.get("class_distribution", {})),
                                  figsize=(5 * len(report["class_distribution"]), 4))
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]

        for ax, (split, dist) in zip(axes, report["class_distribution"].items()):
            counts = [dist[str(i)]["count"] if str(i) in dist else dist.get(i, {}).get("count", 0)
                      for i in range(5)]
            bars = ax.bar(CLASS_NAMES, counts, color=COLORS)
            ax.set_title(f"Class Distribution — {split}")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        str(count), ha="center", va="bottom", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("APTOS Dataset — DR Grade Distribution", fontsize=12)
        plt.tight_layout()
        out_path = results_dir / "class_distribution.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Class distribution chart saved to: {out_path}")

    except Exception as e:
        logger.warning("Could not plot class distribution: %s", e)


if __name__ == "__main__":
    main()
