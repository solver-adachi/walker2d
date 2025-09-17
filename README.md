# Walker2D with Dagger + Stable-Baselines3

このリポジトリは **Dagger** を用いて [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) の  
Walker2D (PyBullet) を学習・再生できる環境です。  
Docker を直接触らなくても、Dagger のモジュールとして再現可能になっています。

## セットアップ

依存は Dagger だけでOKです。Daggerはインストール済みで実行可能であること。

```bash
curl -fsSL https://dl.dagger.io/dagger/install.sh | BIN_DIR=$HOME/.local/bin sh
```

## 学習 (Training)

ステップ学習して `w2d_fresh1/` に保存:

```bash
dagger call -m . train \
  --source . \
  --total-steps 50000000 \
  --n-envs 4 \
  --seed 42 \
  --logdir artifacts/w2d_fresh1 \
  export --path ./artifacts/w2d_fresh1
```

出力:

- `w2d_fresh1/best_model.zip`
- `w2d_fresh1/vecnormalize.pkl`
- チェックポイント (`ckpt_w2d_xxx_steps.zip`)
- `w2d_fresh1/eval.csv` (評価時のリワード推移)

`eval.csv` は `timesteps, mean_reward` の2列を持つシンプルな CSV で、学習の進み具合をスプレッドシートなどで手軽に確認できます。報酬が十分に上がらない場合は `--total-steps` を増やしたり `--n-envs` で並列環境数を増やすと学習が安定しやすくなります。

続き学習する場合は以下を実行する。

```bash
dagger call -m . train \
  --source . \
  --total-steps 200000 \
  --n-envs 4 \
  --seed 42 \
  --logdir artifacts/w2d_fresh1 \
  --resume 1 \
  export --path ./artifacts/w2d_fresh1
```

## 評価結果の確認

安定歩行には必要なreward値は 1,000 前後です。

``` python
python - <<'PY'
import os, glob, numpy as np, pandas as pd
base="artifacts/w2d_fresh1"
cands=[
    os.path.join(base,"eval","evaluations.npz"),
    os.path.join(base,"evaluations.npz"),
]
p=next((x for x in cands if os.path.exists(x)), None)
if p is None:
    # 念のため再帰探索
    g=glob.glob(os.path.join(base,"**","evaluations.npz"), recursive=True)
    if g: p=g[0]
if p is None:
    raise FileNotFoundError("evaluations.npz が見つかりませんでした")
print("[use]", p)
data=np.load(p)
df=pd.DataFrame({"timesteps": data["timesteps"], "mean_reward": data["results"].mean(axis=1)})
print(df.tail())
out=os.path.join(os.path.dirname(p), "eval.csv")
df.to_csv(out, index=False)
print("[saved]", out)
PY
```

## 再生 (Enjoy / Rollout)

学習済みモデルを使って動画(mp4)を生成:

```bash
dagger call -m . enjoy \
  --source . \
  --logdir artifacts/w2d_fresh1 \
  --vecnorm artifacts/w2d_fresh1/vecnormalize.pkl \
  --episodes 3 --det --mu 0.9 --nudge --sleep 0.01 \
  --record artifacts/w2d_fresh1/videos/demo.mp4 \
  export --path ./artifacts/demo.mp4
```

出力:
`./artifacts/demo.mp4` に歩行動画が保存されます

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
- ログと学習曲線は `tensorboard --logdir artifacts/w2d_fresh1` で可視化可能
- 実行中の詳細ログを見たいときは `DAGGER_LOG_LEVEL=debug DAGGER_PROGRESS=plain dagger call ...` のように環境変数で指定すると CLI に内部ログが表示されます

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

学習済みモデルでの歩行例:`artifacts/demo.mp4` を参照

## pythonでの実行手順

セットアップ

```bash
sudo apt update
sudo apt install -y python3.10-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

学習

```bash
python sb3_train_walker2d.py
```

再生

```bash
python sb3_enjoy_walker2d.py
```

終了

```bash
deactivate
```
