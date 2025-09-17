#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import random
import numpy as np
import gymnasium as gym
import pybullet_envs_gymnasium  # ← これが環境登録のトリガ
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
)

ENV_ID = "Walker2DBulletEnv-v0"


# ------------------------------
# Utils
# ------------------------------
def set_headless_env():
    # ヘッドレス実行のための環境変数
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.pop("PYBULLET_EGL", None)


def make_logger(folder: str):
    os.makedirs(folder, exist_ok=True)
    logger = configure(folder, ["csv", "tensorboard", "stdout"])
    # progress.csv を確実に生成
    logger.record("meta/run_id", int(time.time()))
    logger.dump(step=0)
    return logger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=50_000_000)
    ap.add_argument("--logdir", type=str, default="runs_walker2d")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--save-every", type=int, default=100_000)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--resume", action="store_true")  # Dagger 側は bool フラグ運用
    return ap.parse_args()


# ------------------------------
# VecEnv helpers
# ------------------------------
def make_raw_vec(env_id: str, n_envs: int, seed: int, render: bool = False):
    def _fn(i=0):
        e = gym.make(env_id, render_mode=("human" if render else None))
        e.reset(seed=seed + i)
        return Monitor(e)  # 評価メトリクスが安定
    return DummyVecEnv([_fn for _ in range(n_envs)])


def load_or_new_vecnorm(stats_path: str, raw_vec: DummyVecEnv, training: bool):
    """
    既存の VecNormalize 統計があればロード、無ければ新規作成。
    学習用(training=True)・評価用(training=False)どちらも同じ統計を使う。
    """
    if os.path.exists(stats_path):
        v = VecNormalize.load(stats_path, raw_vec)
        print(f"[vecnorm] loaded: {stats_path} (training={training})")
    else:
        v = VecNormalize(
            raw_vec,
            training=training,
            norm_obs=True,
            norm_reward=training,
            clip_obs=10.0,
            gamma=0.99,
        )
        print(f"[vecnorm] new stats (training={training})")
    v.training = training
    v.norm_reward = training
    return v


def build_eval_cb(eval_env, eval_dir: str, eval_freq: int):
    os.makedirs(eval_dir, exist_ok=True)
    return EvalCallback(
        eval_env,
        best_model_save_path=eval_dir,  # <logdir>/eval/ に保存（上書き回避）
        log_path=eval_dir,
        eval_freq=int(max(1, eval_freq)),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )


# ------------------------------
# Main
# ------------------------------
def main():
    args = parse_args()
    if args.no_render:
        set_headless_env()

    # 再現性
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ログ/統計パス
    logdir = args.logdir
    eval_dir = os.path.join(logdir, "eval")
    stats_path = os.path.join(logdir, "vecnormalize.pkl")
    os.makedirs(logdir, exist_ok=True)

    # ===== Env =====
    raw_train = make_raw_vec(ENV_ID, n_envs=args.n_envs, seed=args.seed, render=False)
    train_env = load_or_new_vecnorm(stats_path, raw_train, training=True)

    raw_eval = make_raw_vec(ENV_ID, n_envs=4, seed=args.seed + 123, render=False)
    eval_env = load_or_new_vecnorm(stats_path, raw_eval, training=False)

    # ===== Policy / Model =====
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ortho_init=True,
    )

    # 追い学習でも「安全ハイパラ」（崩壊しにくい）
    def build_fresh_model():
        return PPO(
            "MlpPolicy",
            train_env,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=1e-4,   # 安全：固定で小さめ
            clip_range=0.2,       # クリップ狭め
            ent_coef=0.005,       # 探索弱め
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,       # 過更新ストッパー
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device="auto",
            tensorboard_log=logdir,
        )

    model = None
    reset_num = True

    if args.resume:
        # 同じ logdir の last/best を自動探索
        cand = [
            os.path.join(logdir, "last_model.zip"),
            os.path.join(logdir, "best_model.zip"),
            os.path.join(eval_dir, "best_model.zip"),
        ]
        ckpt = next((p for p in cand if os.path.exists(p)), None)
        if ckpt:
            print(f"[resume] load {ckpt}")
            model = PPO.load(ckpt, env=train_env, device="auto")
            model.set_env(train_env)
            print("[resume] prev timesteps =", getattr(model, "num_timesteps", -1))
            reset_num = False

    if model is None:
        model = build_fresh_model()

    # ===== Logger / Callbacks =====
    logger = make_logger(logdir)
    model.set_logger(logger)

    # n_envs 並列ぶんだけ学習カウンタが進むので、目安として割る
    eval_every = max(1, args.eval_every // max(1, args.n_envs))
    save_every = max(1, args.save_every // max(1, args.n_envs))

    eval_cb = build_eval_cb(eval_env, eval_dir, eval_every)
    ckpt_cb = CheckpointCallback(
        save_freq=save_every,
        save_path=logdir,
        name_prefix="ckpt_w2d",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # ===== Train =====
    total_steps = int(args.total_steps)
    print(f"[train] total_steps={total_steps:,} logdir={logdir} n_envs={args.n_envs} resume={args.resume}")
    model.learn(
        total_timesteps=total_steps,
        reset_num_timesteps=reset_num,
        progress_bar=True,
        callback=[eval_cb, ckpt_cb],
    )

    # ===== Save =====
    last_model_path = os.path.join(logdir, "last_model.zip")
    model.save(last_model_path)
    train_env.save(stats_path)  # VecNormalize の統計を必ず保存
    print(f"[saved] last={last_model_path}")
    if os.path.exists(os.path.join(eval_dir, "best_model.zip")):
        print(f"[saved] best(eval)={os.path.join(eval_dir, 'best_model.zip')}")

    # ===== Export evaluations.npz -> CSV =====
    eval_npz = os.path.join(eval_dir, "evaluations.npz")
    if os.path.exists(eval_npz):
        try:
            data = np.load(eval_npz)
            timesteps = data["timesteps"]
            mean_rewards = data["results"].mean(axis=1)
            out_csv = os.path.join(eval_dir, "eval.csv")
            import csv
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timesteps", "mean_reward"])
                w.writerows(zip(timesteps, mean_rewards))
            print(f"[saved] eval_csv={out_csv}")
        except Exception as e:
            print(f"[warn] failed to export eval csv: {e}")
    else:
        print(f"[warn] {eval_npz} not found; skip eval.csv export")


if __name__ == "__main__":
    main()
