"""
RetinaGuard — EXP-002 Suite Runner
===================================
Executes EXP-002-A, EXP-002-B, and EXP-002-C in sequence.
Collects and prints the exact comparison table, per-class metrics,
and detailed Grade 4 breakdowns as requested by the user.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from training.experiments import run_experiment, ExperimentTracker

def run_suite():
    print("=" * 75)
    print("  STARTING EXP-002 CONTROLLED EXPERIMENT SUITE")
    print("=" * 75)

    tracker_path = PROJECT_ROOT / "results" / "experiments_log.json"
    completed_ids = set()
    if tracker_path.exists():
        try:
            with open(tracker_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                completed_ids = {e["exp_id"] for e in data}
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # EXP-002-A: OrdinalFocalLoss + Standard Sampler
    # ──────────────────────────────────────────────────────────────────────────
    if "EXP-002-A" in completed_ids and (PROJECT_ROOT / "models" / "experiments" / "EXP-002-A" / "best_model.pt").exists():
        print("\n>>> Skipping EXP-002-A: Already completed and logged in experiments_log.json")
    else:
        print("\n>>> Launching EXP-002-A: OrdinalFocalLoss + Standard Sampler")
        run_experiment(
            exp_id="EXP-002-A",
            loss_type="ordinal_focal",
            use_weighted_sampler=False,
            stage1_epochs=3,
            stage2_epochs=0,
            lr_stage1=1e-3,
            gamma=2.0,
            ordinal_lambda=0.25,
            vertical_flip=True,
            rotation_degrees=15,
            batch_size=32,
            seed=42
        )

    # ──────────────────────────────────────────────────────────────────────────
    # EXP-002-B: OrdinalFocalLoss + Moderate Sampler (Inverse Sqrt Frequency)
    # ──────────────────────────────────────────────────────────────────────────
    if "EXP-002-B" in completed_ids and (PROJECT_ROOT / "models" / "experiments" / "EXP-002-B" / "best_model.pt").exists() and (PROJECT_ROOT / "models" / "experiments" / "EXP-002-B" / "metrics.json").exists():
        print("\n>>> Skipping EXP-002-B: Already completed and logged in experiments_log.json")
    else:
        print("\n>>> Launching EXP-002-B: OrdinalFocalLoss + Moderate Weighted Sampler")
        run_experiment(
            exp_id="EXP-002-B",
            loss_type="ordinal_focal",
            use_weighted_sampler=True,
            sampler_power=0.5,       # Moderate square-root inverse frequency
            grade4_boost=1.0,        # No artificial boost
            stage1_epochs=3,
            stage2_epochs=0,
            lr_stage1=1e-3,
            gamma=2.0,
            ordinal_lambda=0.25,
            vertical_flip=True,
            rotation_degrees=15,
            batch_size=32,
            seed=42
        )

    # ──────────────────────────────────────────────────────────────────────────
    # EXP-002-C: OrdinalFocalLoss + Stronger Sampler (Full Inv Freq + 1.5x G4)
    # ──────────────────────────────────────────────────────────────────────────
    if "EXP-002-C" in completed_ids and (PROJECT_ROOT / "models" / "experiments" / "EXP-002-C" / "best_model.pt").exists() and (PROJECT_ROOT / "models" / "experiments" / "EXP-002-C" / "metrics.json").exists():
        print("\n>>> Skipping EXP-002-C: Already completed and logged in experiments_log.json")
    else:
        print("\n>>> Launching EXP-002-C: OrdinalFocalLoss + Strong Weighted Sampler")
        run_experiment(
            exp_id="EXP-002-C",
            loss_type="ordinal_focal",
            use_weighted_sampler=True,
            sampler_power=1.0,       # Full inverse frequency
            grade4_boost=1.5,        # Stronger Grade 4 exposure
            stage1_epochs=3,
            stage2_epochs=0,
            lr_stage1=1e-3,
            gamma=2.0,
            ordinal_lambda=0.25,
            vertical_flip=True,
            rotation_degrees=15,
            batch_size=32,
            seed=42
        )

    print("\n" + "=" * 75)
    print("  ALL EXP-002 EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print("\n>>> Running EXP-002 Evaluation Matrix across all models...")
    from evaluate_exp002_suite import evaluate_all
    evaluate_all()

if __name__ == "__main__":
    run_suite()
