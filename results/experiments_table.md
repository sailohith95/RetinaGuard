# RetinaGuard — Experiment Results Log

| Exp ID | Loss | Sampler | St1/St2 Ep | Acc (%) | QWK | Macro F1 | G0 F1 | G1 F1 | G2 F1 | G3 F1 | G4 F1 | Ref Sens (%) | Ref Spec (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-000 (Baseline) | Weighted CrossEntropy | Standard | 1+0 | 64.5 | 0.6752 | 0.4144 | 0.876 | 0.397 | 0.444 | 0.325 | 0.031 | 67.1 | 94.7 |
| EXP-001 | focal | Standard | 3+0 | 69.8 | 0.7342 | 0.4981 | 0.910 | 0.420 | 0.590 | 0.373 | 0.198 | 78.9 | 92.6 |
| EXP-002-A | ordinal_focal | Standard | 3+0 | 68.8 | 0.6879 | 0.4768 | 0.880 | 0.415 | 0.555 | 0.407 | 0.127 | 69.5 | 96.1 |
| EXP-002-B | ordinal_focal | WeightedRandomSampler | 3+0 | 62.5 | 0.7262 | 0.4090 | 0.904 | 0.419 | 0.277 | 0.382 | 0.062 | 57.7 | 97.7 |
| EXP-002-C | ordinal_focal | WeightedRandomSampler | 3+0 | 54.3 | 0.7214 | 0.3837 | 0.864 | 0.373 | 0.029 | 0.346 | 0.306 | 63.4 | 96.1 |
| EXP-003-A | focal | Standard | 3+3 | 69.8 | 0.7342 | 0.4981 | 0.910 | 0.420 | 0.590 | 0.373 | 0.198 | 78.9 | 92.6 |
| EXP-003-B | focal | Standard | 3+3 | 69.8 | 0.7342 | 0.4981 | 0.910 | 0.420 | 0.590 | 0.373 | 0.198 | 78.9 | 92.6 |
