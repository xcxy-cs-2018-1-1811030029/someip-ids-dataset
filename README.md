# SOME/IP In-Vehicle Intrusion Detection: Dataset + Benchmark

This repository accompanies the paper *"A Reproducible SOME/IP Intrusion-Detection Benchmark: Complementary
Roles of Timing, Byte-Content, and Semantic Features"*. It provides (1) a reproducible, multi-run,
multi-intensity SOME/IP intrusion-detection dataset generated on an open-source vSomeIP testbed, (2) the full
generation pipeline, and (3) a common-protocol benchmark across three detector families (stream behavioral
features, payload semantic-validity, and raw-byte deep sequence models) under random, temporal, and
run-independent splits, including a fixed-false-alarm signal-isolation analysis.

## Dataset

- **189,596 messages**, **57 runs**, **7 scenarios**, issued against a single service/ECU topology.
- Scenarios: normal, DoS, fuzzing, message-drop, low-and-slow, value tampering, contextual tampering.
- Each of the six attack scenarios is generated at **three intensity levels** (`0.3 / 0.7 / 1.0`).
- Per-message records `ts,event,session,payload_len,payload_hex` (+ label/scenario via `manifest.json`).
- The corpus is attack-imbalanced (91.5% positive); the benchmark therefore reports ROC-AUC, PR-AUC, MCC, and
  a fixed-false-alarm TPR as the primary metrics, with F1 reported for completeness.

## Repository Layout

```
someIP/
├── README.md                       # this file
├── vs_ip_gen/
│   ├── setup_vsomeip.sh            # one-click vSomeIP + Boost 1.83 + python deps (WSL/Ubuntu)
│   ├── generate/                   # signal service/client + build/generate/verify scripts
│   │   ├── signal_service.cpp      # generator (multiple attack modes + intensity scaling)
│   │   ├── signal_client.cpp
│   │   ├── build_samples.sh        # compile the generator
│   │   ├── generate_data.sh        # multi-run, multi-intensity generation
│   │   ├── check_attacks.py        # verify attack signatures actually reach the client
│   │   ├── regen_fuzz.sh           # regenerate fuzz at a benign cadence (pure payload attack)
│   │   └── <scenario>_i<int>_run<N>_cli.csv   # the labeled dataset
│   └── README_dataset.md           # dataset documentation (schema, scenarios, labels)
├── someip_ids/                     # evaluation code
│   ├── benchmark.py                # benchmark (random/temporal/run splits; GBM/RF/semantic/GRU/TCN/fused)
│   ├── semantics.py                # payload semantic-validity modeling
│   ├── csv_to_records.py           # CSV logs -> feature arrays
│   └── ...
└── figures/plot_figures.py         # regenerate the vector figures
```

## Reproduce the Dataset

Requires WSL2 Ubuntu-22.04 (NVIDIA GPU optional for training). From `vs_ip_gen/`:

```bash
bash setup_vsomeip.sh                 # build vSomeIP (+ Boost 1.83) and python deps
cd generate
bash build_samples.sh                 # compile signal_service / signal_client
INTENSITIES="0.3 0.7 1.0" RUNS=3 bash generate_data.sh   # multi-run + multi-intensity CSVs + manifest.json
python3 check_attacks.py .            # VERIFY attack signatures are present (no "NO (broken)")
```

To scale, adjust `INTENSITIES`, `RUNS`, `RUNTIME`, and `CYCLE` in `generate_data.sh`.

## Run the Benchmark

```bash
bash vs_ip_gen/generate/run_all.sh    # runs random + temporal + run-independent splits; saves to results/
```

or per split (needs PyTorch for GRU/TCN):

```bash
python3 someip_ids/benchmark.py --gen vs_ip_gen/generate/ --split run     --epochs 30 --fpr-eval
python3 someip_ids/benchmark.py --gen vs_ip_gen/generate/ --split random  --epochs 30 --fpr-eval
python3 someip_ids/benchmark.py --gen vs_ip_gen/generate/ --split temporal --epochs 30 --fpr-eval
```

Reports overall + per-attack metrics for GBM, RandomForest, semantic-validity, GRU, TCN, and a fused
detector under each split. `--fpr-eval` adds the fixed-1%-FPR signal-isolation analysis and `--diagnose`
adds the timing-only / byte-only / protocol-only feature-family ablation.

## Companion Manuscript

The accompanying paper is a separate submission and is not bundled in this repository; the dataset, generator,
and evaluation code here reproduce its results.

## License & Data Release

Dataset, generator, and evaluation code are released publicly for reproducible research at
**https://github.com/xcxy-cs-2018-1-1811030029/someip-ids-dataset**. See `vs_ip_gen/README_dataset.md` for the
dataset specification.
