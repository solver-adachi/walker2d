# src/walker2d/mod_impl.py
from dagger import dag, object_type, function, Directory, File

BASE_IMAGE = "python:3.10-slim"
WORKDIR = "/app"

APT = [
    "libgl1", "libglx-mesa0", "libegl1", "libgl1-mesa-dri",
    "libosmesa6-dev",   # ★ OSMesa を追加
    "libglib2.0-0", "ffmpeg", "ca-certificates",
    "xvfb",            # ← 追加
    "libosmesa6",      # ← 追加（保険：OSMesa ソフトレンダ）
]
PIP = [
    "numpy<2",
    "gymnasium==0.29.1",
    "pybullet==3.2.6",
    "pybullet_envs_gymnasium==0.6.0",
    "stable-baselines3[extra]==2.3.2",  # ← extras を明示
    "imageio==2.37.0",
    "imageio-ffmpeg==0.4.9",
    "tensorboard==2.17.1",
]

@object_type
class Walker2D:
    def _base(self, source: Directory):
        return (
            dag.container()
            .from_(BASE_IMAGE)
            .with_env_variable("PYTHONUNBUFFERED", "1")
            .with_env_variable("PYTHONHASHSEED", "0")
            .with_env_variable("SDL_VIDEODRIVER", "dummy")
            .with_env_variable("PYOPENGL_PLATFORM", "osmesa")
            .with_workdir(WORKDIR)
            .with_exec([
                "bash","-lc",
                "apt-get update && apt-get install -y --no-install-recommends "
                "libosmesa6 libosmesa6-dev xvfb xauth x11-utils "
                "libgl1 libglx-mesa0 libegl1 libgl1-mesa-dri "
                "libglib2.0-0 ffmpeg ca-certificates && rm -rf /var/lib/apt/lists/*"
            ])
            .with_exec(["bash","-lc","python -m pip install --upgrade pip setuptools wheel"])
            .with_exec(["pip","install","--no-cache-dir", *PIP])
            .with_mounted_directory(WORKDIR, source)
            .with_workdir(WORKDIR)
            # 環境スナップショットを残す
            .with_exec(["bash","-lc",
                "python - <<'PY'\n"
                "import sys, pkgutil, torch\n"
                "print('python',sys.version)\n"
                "print('torch', torch.__version__)\n"
                "import pkg_resources as pr\n"
                "for p in sorted([d for d in pr.working_set], key=lambda x:x.project_name.lower()):\n"
                "    print(f'{p.project_name}=={p.version}')\n"
                "PY\n"
                "> env.txt || true"])
        )

    @function
    async def enjoy(
        self,
        source: Directory,
        episodes: int = 1,
        sleep: float = 0.006,
        det: bool = True,
        mu: float | None = None,
        nudge: bool = False,
        no_render: bool = False,
        record: str | None = "out.mp4",
        # ★ ここが無いと `--logdir/--vecnorm` が unknown flag になります
        logdir: str = "runs_w2d_dagger",
        vecnorm: str = "runs_w2d_dagger/vecnormalize.pkl",
    ) -> File:
        ctr = self._base(source)
        ctr = ctr.with_exec([
            "bash","-lc",
            f"mkdir -p {logdir}/videos; "  # 既存
            f"mkdir -p $(dirname {record})"  # ← これを追加
        ])
        # モデル / VecNorm の存在チェック（無ければ明示エラーにする）
        check = (
            "set -e; "
            f"echo '[ls] {logdir}:'; ls -l {logdir} || true; "
            f"test -f {vecnorm} && echo '[check] vecnorm: OK' || (echo '[check] vecnorm: MISSING' && exit 2); "
            "echo -n '[check] model: '; "
            f"if   [ -f {logdir}/last_model.zip ]; then echo {logdir}/last_model.zip; "
            f"elif [ -f {logdir}/best_model.zip ]; then echo {logdir}/best_model.zip; "
            f"elif [ -f {logdir}/eval/best_model.zip ]; then echo {logdir}/eval/best_model.zip; "
            f"elif [ -f {logdir}/ppo_walker2d_sb3.zip ]; then echo {logdir}/ppo_walker2d_sb3.zip; "
            "else echo MISSING && exit 3; fi"
        )
        ctr = ctr.with_exec(["bash","-lc", check])
        ctr = ctr.with_exec([
            "bash","-lc",
            f"sha256sum {logdir}/best_model.zip 2>/dev/null || true; "
            f"sha256sum {vecnorm} 2>/dev/null || true"
        ])
        # 録画指定時は内部レンダ必須（human表示はしない）
        if record and no_render:
            no_render = False

        args = [
            "python", "sb3_enjoy_walker2d.py",
            "--logdir", logdir,
            "--vecnorm", vecnorm,
        ]
        # mu は指定があるときだけ渡す（デフォルトでは渡さない）
        if mu is not None:
           args += ["--mu", str(mu)]
        if det:       args.append("--det")
        if nudge:     args.append("--nudge")
        if no_render: args.append("--no-render")
        if record:    args += ["--record", record]

        # ログ出し & 実行
        run = " ".join(args)
        ctr = ctr.with_exec([
            "bash","-lc",
            f"printf '[run] %q ' xvfb-run -s \"-screen 0 1280x720x24\" -a {run}; echo"
        ])
        ctr = ctr.with_exec([
            "bash","-lc",
            f"PYBULLET_EGL=0 LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -s \"-screen 0 1280x720x24\" -a {run}"
        ])

        return ctr.file(f"{WORKDIR}/{record or 'out.mp4'}")

    @function
    async def train(
        self,
        source: Directory,
        total_steps: int = 2_000_000,
        logdir: str = "artifacts/runs_w2d_dagger",
        seed: int = 42,
        n_envs: int = 1,
        resume: int = 1,  # 追加：0=新規/上書きなし, 1=続き学習
    ) -> Directory:
        ctr = self._base(source)
        
        # ログディレクトリ準備（resume=0なら中身を空にしてから）
        if resume == 0:
            ctr = ctr.with_exec(["bash","-lc", f"rm -rf {logdir} && mkdir -p {logdir}"])
        else:
            ctr = ctr.with_exec(["bash","-lc", f"mkdir -p {logdir}"])
        
        ctr = ctr.with_exec(["bash","-lc","mkdir -p " + logdir])
        print("CMD:", [
            "python","-u","sb3_train_walker2d.py",
            "--total-steps", str(total_steps),
            "--logdir", logdir,
            "--seed", str(seed),
            "--n-envs", str(n_envs),
            "--eval-every","50000",
            "--save-every","100000",
            "--no-render",
            ] + (["--resume"] if resume else []))
            # 進捗バーやTBは extras 済み。実行前にtorch deterministic化
        ctr = ctr.with_exec(["bash","-lc",
            "python - <<'PY'\n"
            "import torch\n"
            "torch.use_deterministic_algorithms(True)\n"
            "print('[diag] torch deterministic ON')\n"
            "PY"])
        # 学習
        args = [
            "python","-u","/app/sb3_train_walker2d.py",
            "--total-steps", str(total_steps),
            "--logdir", logdir,
            "--seed", str(seed),
            "--n-envs", str(n_envs),
            "--eval-every","50000",
            "--save-every","100000",
            "--no-render",
        ]
        if resume:          # ← Dagger側の bool/int を判定
            args.append("--resume")   # ← 値なし（store_true）

        ctr = ctr.with_exec(args)
        return ctr.directory(f"{WORKDIR}/{logdir}")
