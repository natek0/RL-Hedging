#!/usr/bin/env bash
# Train seeds 1 and 2 for every learned policy (seed 0 comes from run_all.sh), two jobs at a time,
# then re-evaluate. On an 8-core laptop raise the parallelism (e.g. xargs -P 8).
set -euo pipefail
cd "$(dirname "$0")"
jobs=()
for s in 1 2; do for p in 1 2; do
  jobs+=("python scripts/train_rl.py --algo ppo --phase $p --seed $s --steps 500000")
  jobs+=("python scripts/train_rl.py --algo sac --phase $p --seed $s --steps 200000 --n-envs 4")
  jobs+=("python scripts/train_deephedge.py --phase $p --seed $s")
done; done
printf '%s\n' "${jobs[@]}" | xargs -P "${NPROC:-2}" -I{} bash -c '{} > /dev/null && echo done: {}'
for p in 1 2; do
  python scripts/run_eval.py --phase $p --cost 0.002 --kappa 1
  python scripts/make_figures.py --phase $p --cost 0.002 --kappa 1
done
