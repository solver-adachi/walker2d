#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import gymnasium as gym
import pybullet as p
import pybullet_envs_gymnasium  # register Bullet envs
import imageio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ENV_ID = "Walker2DBulletEnv-v0"

# -------------------------
# Headless / render utils
# -------------------------
def set_headless_env():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("PYBULLET_EGL", "1")

def make_env(render=False, record=False):
    # 録画時は必ず rgb_array（human はヘッドレスで死ぬ/None はフレーム取れない）
    mode = "rgb_array" if record else ("human" if render else None)
    return gym.make(ENV_ID, render_mode=mode)

def load_env(vecnorm_path: str, render: bool, record: bool):
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = DummyVecEnv([lambda: make_env(render=render, record=record)])
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"[INFO] VecNormalize loaded: {vecnorm_path}")
        return venv, True
    else:
        print("[WARN] VecNormalize not used")
        return make_env(render=render, record=record), False

def _unwrap_base_env(env_like):
    base = env_like
    if hasattr(base, "venv"): base = base.venv        # VecNormalize 内側
    if hasattr(base, "envs"): base = base.envs[0]      # DummyVecEnv[0]
    while hasattr(base, "env"): base = base.env        # ラッパを剥がす
    return getattr(base, "unwrapped", base)

def grab_frame(env_like):
    base = _unwrap_base_env(env_like)
    try:
        return base.render()
    except Exception:
        return None

# -------------------------
# Bullet helpers（簡易版）
# -------------------------
def _find_bullet_client(base):
    for attr in ["_p", "client", "bullet_client", "pybullet_client"]:
        bc = getattr(base, attr, None)
        if bc and all(hasattr(bc, m) for m in ["getNumBodies","getBodyInfo","changeDynamics"]):
            return bc
    for holder, attr in [("scene", "_p"), ("robot", "_p")]:
        obj = getattr(base, holder, None)
        bc = getattr(obj, attr, None) if obj is not None else None
        if bc and all(hasattr(bc, m) for m in ["getNumBodies","getBodyInfo","changeDynamics"]):
            return bc
    return None

def friction_patch(env_like, mu=None):
    if mu is None: return
    try:
        base = _unwrap_base_env(env_like)
        bc = _find_bullet_client(base)
        if bc is None: return
        patched = False
        for bid in range(bc.getNumBodies()):
            try:
                name = bc.getBodyInfo(bid)[1].decode("utf-8")
            except Exception:
                name = ""
            if "plane" in name.lower():
                for lid in range(bc.getNumJoints(bid) + 1):
                    bc.changeDynamics(bid, lid, lateralFriction=float(mu))
                patched = True
                break
        if not patched:
            for bid in range(bc.getNumBodies()):
                for lid in range(bc.getNumJoints(bid) + 1):
                    bc.changeDynamics(bid, lid, lateralFriction=float(mu))
        try:
            bc.setPhysicsEngineParameter(enableConeFriction=1, numSolverIterations=80)
        except Exception:
            pass
        print(f"[INFO] friction patched: mu={mu}")
    except Exception as e:
        print("[WARN] friction patch failed:", e)

def nudge_once(env_like, force=40.0):
    try:
        base = _unwrap_base_env(env_like)
        bc = _find_bullet_client(base)
        if bc is None: return
        for bid in range(bc.getNumBodies()):
            if bc.getNumJoints(bid) > 0:
                bc.applyExternalForce(bid, -1, [force,0,0], [0,0,1.0], bc.WORLD_FRAME)
                print(f"[diag] nudge Fx={force}N on body {bid}")
                break
    except Exception as e:
        print("[WARN] nudge failed:", e)

# -------------------------
# Step / Reset helpers
# -------------------------
def step_compat(env_like, action, is_vec):
    out = env_like.step(action)
    if is_vec:
        obs, r, done, info = out           # SB3 VecEnv 形式
        r = float(np.asarray(r).mean())    # n_envs=1 なので平均でOK
        done = bool(done[0])               # ← これが重要
        return obs, r, done
    # Gymnasium 形式
    if isinstance(out, tuple) and len(out) == 5:
        obs, r, terminated, truncated, info = out
        return obs, float(r), bool(terminated or truncated)
    obs, r, done, info = out
    return obs, float(r), bool(done)

def reset_obs(env, is_vec):
    out = env.reset()
    return out if is_vec else out[0]  # Gymnasiumは (obs, info)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="runs_walker2d")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--vecnorm", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--det", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=None)
    ap.add_argument("--nudge", action="store_true")
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--fps", type=int, default=166)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    # seeds
    if args.seed is not None:
        try:
            import random, torch
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        except Exception:
            try:
                import random
                random.seed(args.seed)
                np.random.seed(args.seed)
            except Exception:
                pass

    # ヘッドレス条件
    recording = bool(args.record)
    if args.no_render or recording:
        set_headless_env()

    # human表示は録画時にOFF（二重描画を避ける）
    render = (not args.no_render) and (not recording)

    # env 構築 → reset → 物理パッチ/ナッジ（この順番！）
    env, is_vec = load_env(args.vecnorm, render=render, record=recording)
    if is_vec:
        assert isinstance(env, VecNormalize), "[ERR] VecNormalize not applied"
    obs = reset_obs(env, is_vec)
    friction_patch(env, mu=args.mu)
    if args.nudge:
        nudge_once(env, force=40.0)

    # モデル
    model_path = args.model or os.path.join(args.logdir, "best_model.zip")
    if not os.path.exists(model_path):
        alt = os.path.join(args.logdir, "ppo_walker2d_sb3.zip")
        if os.path.exists(alt): model_path = alt
    print(f"[INFO] loading model: {model_path}")
    model = PPO.load(model_path)

    # writer 準備
    writer = None
    if recording:
        fps = max(1, min(240, (args.fps if args.fps > 0 else (int(1/args.sleep) if args.sleep > 0 else 30))))
        writer = imageio.get_writer(args.record, fps=fps)

    # 本編
    for ep in range(1, args.episodes + 1):
        ret, steps, done = 0.0, 0, False
        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=args.det)
            obs, r, done = step_compat(env, action, is_vec)
            ret += r; steps += 1

            if recording:
                frame = grab_frame(env)
                if frame is not None:
                    f = np.asarray(frame)
                    if f.dtype != np.uint8:
                        f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
                    writer.append_data(f)

            if render and args.sleep > 0:
                time.sleep(args.sleep)

        print(f"[EP DONE] ep={ep}/{args.episodes} return={ret:.1f} steps={steps}")

        if ep < args.episodes:
            obs = reset_obs(env, is_vec)
            friction_patch(env, mu=args.mu)
            if args.nudge:
                nudge_once(env, force=40.0)

    if writer is not None:
        writer.close()
        print(f"[REC] video saved to {args.record}")

    try:
        env.close()
        base = _unwrap_base_env(env)
        bc = _find_bullet_client(base)
        if bc: p.disconnect(physicsClientId=getattr(bc, "_client", None))
    except Exception:
        pass

if __name__ == "__main__":
    main()
