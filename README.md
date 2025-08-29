# Walker2D with Dagger + Stable-Baselines3

このリポジトリは **Dagger** を用いて [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) の  
Walker2D (PyBullet) を学習・再生できる環境です。  
Docker を直接触らなくても、Dagger のモジュールとして再現可能になっています。

## セットアップ

依存は Dagger だけでOKです。

```bash
# dagger CLI のインストール
curl -fsSL https://dl.dagger.io/dagger/install.sh | sh
```

## 学習 (Training)

ステップ学習して `runs_w2d_dagger/` に保存:

```bash
dagger call -m . train \
  --source . \
  --total-steps 20000000 \
  --seed 42 \
  --logdir artifacts/runs_w2d_dagger \
  export --path ./artifacts/runs_w2d_dagger
```

出力:

- `runs_w2d_dagger/best_model.zip`
- `runs_w2d_dagger/vecnormalize.pkl`
- チェックポイント (`ckpt_w2d_xxx_steps.zip`)

## 再生 (Enjoy / Rollout)

学習済みモデルを使って動画(mp4)を生成:

```bash
dagger call -m . enjoy \
  --source . \
  --logdir artifacts/runs_w2d_dagger \
  --vecnorm artifacts/runs_w2d_dagger/vecnormalize.pkl \
  --episodes 3 --det --nudge --mu 0.9 --sleep 0.01 \
  --record out.mp4 \
  export --path ./artifacts/out.mp4
```

出力:
`./artifacts/out.mp4` に歩行動画が保存されます

## オプション引数

- `--episodes N` : 評価するエピソード数 (既定=3)
- `--sleep SEC` : 1ステップごとのスリープ (動画のfps調整用)
- `--det` : 確定的 (deterministic) 行動を使用
- `--nudge` : 開始時に外力を与えて姿勢を乱す
- `--mu` : 摩擦係数を上書き (ドメインランダム化用)
- `--record FILENAME` : 動画を保存 (mp4形式)

## Tips

- CI/CD環境(GitLab Runnerなど)でも `dagger call` を叩けば学習と動画生成が自動化可能
- ドメインランダム化: `--mu` をランダム化して学習・評価すると汎化性能UP
- ログと学習曲線は `tensorboard --logdir runs_w2d_dagger` で可視化可能

## ディレクトリ構成

```bash
walker2d/
 ├─ pyproject.toml             # project定義
 ├─ dagger.json                # Daggerモジュール定義
 ├─ src/walker2d/__init__.py
 ├─ src/walker2d/mod_impl.py   # train/enjoy 実装
 ├─ sb3_train_walker2d.py
 ├─ sb3_enjoy_walker2d.py
 └─ artifacts/                 # 出力 (動画, 学習成果物)
```

## 動画サンプル

学習済みモデルでの歩行例:`artifacts/out.mp4` を参照
