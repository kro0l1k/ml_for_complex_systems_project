"""
ppo_control.py
Train a PPO agent to learn the feedback-control u* for the 1-D jump–diffusion
portfolio problem defined in config.py (see original script).

Dependencies
------------
pip install gymnasium torch stable-baselines3[extra]
"""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ----------------------------------------------
# 1. Gymnasium environment
# ----------------------------------------------
class JumpDiffusionEnv(gym.Env):
    """
    Observation  = [t,  X_t]               (shape = (2,))
    Action       = u_t   ∈ ℝ               (1-D Box)
    Reward       = −0.5·(X_T − a)^2        (only at terminal step)
    Episode len  = config.time_step_count  (≈200)
    """

    metadata = {"render_modes": ["none"]}

    def __init__(self, config, render_mode="none"):
        super().__init__()
        self.cfg = config
        self.render_mode = render_mode

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -np.inf]),
            high=np.array([self.cfg.terminal_time, np.inf]),
            dtype=np.float32,
        )
        # Put *reasonable* bounds on u_t so PPO doesn’t diverge
        self.action_space = gym.spaces.Box(
            low=np.array([-5.0]), high=np.array([5.0]), dtype=np.float32
        )

        self.t_idx = 0
        self.X = self.cfg.X_init.clone()          # shape (1,)
        self.sample = None                        # will hold one  path of dW & jumps

    # ---------- helpers -----------------------------------------------------
    def _sample_noise(self):
        """One Monte-Carlo path for a single episode"""
        s = self.cfg.sample(1)                    # sample_size=1
        # Squeeze batch dimension
        s = s._replace(
            delta_W_TBW=s.delta_W_TBW[:, 0],
            jump_mask_TBLC=s.jump_mask_TBLC[:, 0],
            jump_sizes_BLCX=s.jump_sizes_BLCX[0],
        )
        return s

    # ---------- Gym API ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t_idx = 0
        self.X = self.cfg.X_init.clone()          # tensor (1,)
        self.sample = self._sample_noise()

        obs = np.array([0.0, self.X.item()], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Propagate one Δt step with control u = action[0]
        """
        u = torch.tensor(action, dtype=torch.float32, device=self.X.device)
        if u.ndim == 1:          # PPO gives shape (1,) here
            u = u.unsqueeze(0)   # -> (1,1)
       
        i = self.t_idx

        # Ito + jumps (copy of your Euler scheme, but for a *single* sample)
        self.X = (
            self.X
            + self.cfg.drift(0.0, self.X, u) * self.cfg.delta_t
            + (self.cfg.diffusion(0.0, self.X, u)[:, 0]  # dim_W=1
               * self.sample.delta_W_TBW[i])
            + self.X
            * (self.sample.jump_mask_TBLC[i]              # (dim_L,max_c)
               * self.sample.jump_sizes_BLCX).sum()
            - self.cfg.jump_intensity[0]
            * self.cfg.jump_size_mean[0]
            * self.X
            * self.cfg.delta_t
        )

        self.t_idx += 1
        done = self.t_idx >= self.cfg.time_step_count
        truncated = False

        if done:
            reward = (
                -0.5 * (self.X - self.cfg.a_in_cost()) ** 2
            ).item()  # scalar float
        else:
            reward = 0.0

        obs = np.array(
            [self.t_idx * self.cfg.delta_t, self.X.item()], dtype=np.float32
        )
        info = {}
        return obs, reward, done, truncated, info

    def render(self):
        pass  # no-op


# ----------------------------------------------
# 2. PPO trainer wrapper
# ----------------------------------------------
class PPOAgent:
    def __init__(self, config, hyperparams: Dict = None, log_dir="ppo_logs"):
        self.cfg = config
        self.hparams = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 256,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "target_kl": None,
            "policy_kwargs": dict(net_arch=[64, 64]),
        }
        if hyperparams:
            self.hparams.update(hyperparams)

        # Vectorised + normalised env speeds up training
        def make_env():
            return JumpDiffusionEnv(self.cfg)

        venv = DummyVecEnv([make_env])
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

        self.model = PPO(env=venv, verbose=1, tensorboard_log=log_dir, **self.hparams)

    # ---------- public API ---------------------------------------------------
    def train(self, timesteps: int = 200_000):
        self.model.learn(total_timesteps=timesteps)

    def save(self, path: str | Path):
        self.model.save(path)

    @classmethod
    def load(cls, path: str | Path, config):
        loaded = PPO.load(path)
        agent = cls(config)
        agent.model = loaded
        return agent

    def policy(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action for given observation"""
        return self.model.predict(obs, deterministic=True)[0]

    # ---------- evaluation ---------------------------------------------------
    def evaluate_cost(self, eval_paths: int = 1_000):
        """
        Run the trained policy in *open-loop* Monte-Carlo, identical to ClosedFormSolver,
        and return mean / std of cost functional so results are directly comparable.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}, for eval")
        sample_data = self.cfg.sample(eval_paths)
        X = torch.ones(eval_paths, 1, dtype=torch.float32, device=device) * self.cfg.X_init

        for i in range(self.cfg.time_step_count):
            t = i * self.cfg.delta_t
            obs = torch.stack(
                [torch.full_like(X, t), X], dim=2
            ).squeeze()  # shape (paths, 2)
            # PPO is trained on normalised vec env ⇒ don’t reuse that here;
            # we just query policy net directly
            with torch.no_grad():
                action = torch.tensor(
                    self.policy(obs.cpu().numpy()), dtype=torch.float32, device=device
                )
            u = action

            X = (
                X
                + self.cfg.drift(t, X, u) * self.cfg.delta_t
                + torch.einsum(
                    "bxw,bw->bx", self.cfg.diffusion(t, X, u), sample_data.delta_W_TBW[i]
                )
                + X
                * torch.einsum(
                    "blc,blcx->bx",
                    sample_data.jump_mask_TBLC[i],
                    sample_data.jump_sizes_BLCX,
                )
                - self.cfg.jump_intensity[0]
                * self.cfg.jump_size_mean[0]
                * X
                * self.cfg.delta_t
            )

        cost = -0.5 * (X - self.cfg.a_in_cost()) ** 2   # (paths,1)
        mean, std = cost.mean().item(), cost.std().item()
        return mean, std


# ----------------------------------------------
# 3. CLI / quick experiment
# ----------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=250_000, help="PPO timesteps")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--batch", type=int, default=256, help="Minibatch size")
    p.add_argument("--gamma", type=float, default=0.995, help="Discount factor γ")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    from closed_form_sol import Config  # re-use the existing class

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config()
    hps = dict(learning_rate=args.lr, batch_size=args.batch, gamma=args.gamma)

    agent = PPOAgent(config=cfg, hyperparams=hps)
    print("⏳ Training PPO …")
    agent.train(timesteps=args.timesteps)
    agent.save("ppo_jump_diffusion.zip")

    print("✅ training done – evaluating …")
    mean_c, std_c = agent.evaluate_cost(eval_paths=5_000)
    print(f"PPO   : E[-½(X_T−a)^2] = {mean_c:.4f}  ± {std_c:.4f}")

    # For reference you could call ClosedFormSolver here as well

if __name__ == "__main__":
    main()
