"""
RetinaGuard — Model Comparison & Selection (Phase 10)
======================================================
Compares the Baseline model against all experimental models.
Generates side-by-side metric tables and detailed class-level differentials.

Rules:
1. Do not replace baseline unless the candidate model is truly better.
2. Model selection balances QWK, Macro F1, minority recall (Grade 4), and referable sensitivity.
"""

import json
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]

def compare():
    log_file = PROJECT_ROOT / "results" / "experiments_log.json"
    if not log_file.exists():
        print("No experiments logged yet.")
        return

    with open(log_file, "r", encoding="utf-8") as f:
        experiments = json.load(f)

    if not experiments:
        print("Empty experiment log.")
        return

    baseline = next((e for e in experiments if "Baseline" in e["exp_id"]), experiments[0])
    b_m = baseline["metrics"]
    b_pc = b_m["per_class"]

    print("=" * 85)
    print("  RETINAGUARD: EXPERIMENTAL MODEL COMPARISON AGAINST BASELINE")
    print("=" * 85)
    print(f"{'Metric':<25} {'Baseline':<14} " + " ".join([f"{e['exp_id']:<14}" for e in experiments if e != baseline]))
    print("-" * 85)

    def row(label, get_val, fmt="{:.4f}"):
        b_val = get_val(baseline)
        cols = [f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val)]
        for e in experiments:
            if e == baseline:
                continue
            v = get_val(e)
            diff = v - b_val if isinstance(v, (int, float)) and isinstance(b_val, (int, float)) else 0
            diff_str = f" ({diff:+.3f})" if isinstance(v, float) else ""
            cols.append(f"{v:.4f}{diff_str}" if isinstance(v, float) else str(v))
        print(f"{label:<25} " + " ".join([f"{c:<14}" for c in cols]))

    row("QWK (Primary)", lambda e: e["metrics"]["qwk"])
    row("Macro F1", lambda e: e["metrics"]["macro_f1"])
    row("Accuracy", lambda e: e["metrics"]["accuracy"])
    row("Referable Sens (>=G2)", lambda e: e["metrics"]["referable_sensitivity"])
    row("Referable Spec (<G2)", lambda e: e["metrics"]["referable_specificity"])
    print("-" * 85)
    for c in CLASS_NAMES:
        row(f"F1: {c}", lambda e, cls=c: e["metrics"]["per_class"][cls]["f1"])
    print("-" * 85)
    for c in CLASS_NAMES:
        row(f"Recall: {c}", lambda e, cls=c: e["metrics"]["per_class"][cls]["recall"])
    print("=" * 85)

if __name__ == "__main__":
    compare()
