"""Train a stable-baselines3 agent (PPO or SAC) on the Heston training bank."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, N_TRAIN_PATHS, TRAIN_SEED, base_parser, make_cfg, tag  # noqa: E402

from stable_baselines3 import PPO, SAC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor  # noqa: E402

from hedgerl.env import HedgingEnv  # noqa: E402
from hedgerl.sim import generate_paths  # noqa: E402


def make_model(algo: str, venv, seed: int):
    if algo == "ppo":
        return PPO("MlpPolicy", venv, seed=seed, n_steps=1024, batch_size=256, n_epochs=10, learning_rate=3e-4,
                   gamma=1.0, gae_lambda=0.95, ent_coef=0.0, clip_range=0.2,
                   policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), log_std_init=-1.0), verbose=0)
    if algo == "sac":
        return SAC("MlpPolicy", venv, seed=seed, learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                   gamma=1.0, train_freq=1, gradient_steps=1, learning_starts=5_000,
                   policy_kwargs=dict(net_arch=[64, 64]), verbose=0)
    raise ValueError(algo)


if __name__ == "__main__":
    p = base_parser("Train RL hedger")
    p.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-paths", type=int, default=N_TRAIN_PATHS)
    a = p.parse_args()
    torch.set_num_threads(1)
    cfg = make_cfg(a.phase, a.cost, a.kappa)
    bank = generate_paths(cfg, a.n_paths, seed=TRAIN_SEED)
    venv = VecMonitor(DummyVecEnv([lambda i=i: HedgingEnv(cfg, bank, seed=a.seed * 100 + i) for i in range(a.n_envs)]))
    model = make_model(a.algo, venv, a.seed)
    t0 = time.time()
    model.learn(total_timesteps=a.steps, progress_bar=False)
    MODELS.mkdir(exist_ok=True)
    out = MODELS / f"{a.algo}_{tag(a.phase, a.cost, a.kappa)}_s{a.seed}"
    model.save(out)
    print(f"saved {out}.zip  ({time.time() - t0:.0f}s, {a.steps} steps)")
