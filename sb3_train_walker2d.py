#!/usr/bin/env python3
import os, argparse, random
import numpy as np
import gymnasium as gym
import pybullet_envs_gymnasium   # noqa: F401
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement

ENV_ID = "Walker2DBulletEnv-v0"

# --------------------------------------------------
# Utility
# --------------------------------------------------
def set_headless_env():
    # ヘッドレス用 (EGLを明示的に切る)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.pop("PYBULLET_EGL", None)

def make_env(render=False, seed: int = 0):
    mode = None if not render else "human"
    env = gym.make(ENV_ID, render_mode=mode)
    env.reset(seed=seed)
    return env

def build_vec_env(n_envs=1, render=False, seed=42, train=True):
    envs = [lambda i=i: make_env(render=render, seed=seed+i) for i in range(n_envs)]
    venv = DummyVecEnv(envs)
    venv = VecNormalize(
        venv,
        training=train,
        norm_obs=True,
        norm_reward=train,
        clip_obs=10.0,
        gamma=0.99,
    )
    np.random.seed(seed)
    random.seed(seed)
    return venv

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=2_000_000)
    ap.add_argument("--logdir", type=str, default="./runs_w2d_clean")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-envs", type=int, default=1)   # 並列数
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--save-every", type=int, default=100_000)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    return ap.parse_args()

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    if args.no_render:
        set_headless_env()

    # ====== Env ======
    train_env = build_vec_env(n_envs=args.n_envs, render=False, seed=args.seed, train=True)
    eval_env  = build_vec_env(n_envs=1, render=False, seed=args.seed+123, train=False)

    # ★学習用 VecNormalize の統計を評価用にも共有
    eval_env.obs_rms = train_env.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False

    # ====== Model ======
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ortho_init=True,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"[resume] load {args.resume}")
        model = PPO.load(args.resume, device="auto")
        model.set_env(train_env)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=linear_schedule(3e-4),   # ★ 途中で下がる
            clip_range=0.2,
            ent_coef=0.01,        # ★ 最初だけ探索↑ → 0へ
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.015,                       # ★ 暴走ブレーキ
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device="auto",
            tensorboard_log=args.logdir,
        )

    # ====== Callbacks ======
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10, min_evals=5, verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=args.logdir,
        log_path=args.logdir,
        eval_freq=max(1, args.eval_every // max(1, args.n_envs)),
        n_eval_episodes=5,
        deterministic=True,
        callback_after_eval=stop_cb,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_every // max(1, args.n_envs)),
        save_path=args.logdir,
        name_prefix="ckpt_w2d",
    )

    # ====== Train ======
    total_steps = int(args.total_steps)
    print(f"[train] total_steps={total_steps:,} logdir={args.logdir} n_envs={args.n_envs}")
    model.learn(total_timesteps=total_steps, callback=[eval_cb, ckpt_cb], progress_bar=True)

    # ====== Save ======
    last_model_path = os.path.join(args.logdir, "last_model.zip")
    model.save(last_model_path)
    train_env.save(os.path.join(args.logdir, "vecnormalize.pkl"))
    print(f"[saved] last={last_model_path}")

    best_model_path = os.path.join(args.logdir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"[saved] best={best_model_path}")

def linear_schedule(initial_value: float):
    def _thunk(progress_remaining: float):
        return progress_remaining * initial_value
    return _thunk

if __name__ == "__main__":
    main()
