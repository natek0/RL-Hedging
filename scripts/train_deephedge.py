"""Train the Buehler-style deep hedger on the Heston training bank."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, N_TRAIN_PATHS, TRAIN_SEED, base_parser, make_cfg, tag  # noqa: E402

from hedgerl.deephedge import train_deep_hedger  # noqa: E402
from hedgerl.sim import generate_paths  # noqa: E402

if __name__ == "__main__":
    p = base_parser("Train deep hedger")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--n-paths", type=int, default=N_TRAIN_PATHS)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    cfg = make_cfg(a.phase, a.cost, a.kappa)
    bank = generate_paths(cfg, a.n_paths, seed=TRAIN_SEED)
    model = train_deep_hedger(cfg, bank, epochs=a.epochs, seed=a.seed)
    MODELS.mkdir(exist_ok=True)
    out = MODELS / f"deephedge_{tag(a.phase, a.cost, a.kappa)}_s{a.seed}.pt"
    torch.save(model.state_dict(), out)
    print("saved", out)
