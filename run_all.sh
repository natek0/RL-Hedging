#!/usr/bin/env bash
# Reproduce every table and figure in results/. CPU only; ~40 minutes on a 2-core machine.
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests

for PHASE in 1 2; do
  python scripts/train_deephedge.py --phase $PHASE --cost 0.002 --epochs 400 --n-paths 50000 --seed 0
  python scripts/train_rl.py --algo ppo --phase $PHASE --cost 0.002 --steps 500000
  python scripts/train_rl.py --algo sac --phase $PHASE --cost 0.002 --steps 200000 --n-envs 4
  python scripts/run_eval.py --phase $PHASE --cost 0.002 --n-paths 20000 --n-boot 1000
  python scripts/make_figures.py --phase $PHASE --cost 0.002
done
