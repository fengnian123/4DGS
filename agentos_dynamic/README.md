# AgentOS 本地重建流程

## 安装与模型

在仓库根目录执行：

```bash
python3 scripts/bootstrap.py som agentos-v2 agentos-v3
bash scripts/setup.sh som
som_local/.venv/bin/python -m pip install -r agentos_dynamic/requirements-data.txt
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
git -C third_party/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
som_local/.venv/bin/python -m pip install -e third_party/sam2 --no-build-isolation
mkdir -p agentos_dynamic/models
curl -fL https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt -o agentos_dynamic/models/bootstapir_checkpoint_v2.pt
curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -o agentos_dynamic/models/sam2.1_hiera_tiny.pt
som_local/.venv/bin/hf download depth-anything/Depth-Anything-V2-Small-hf --local-dir agentos_dynamic/models/depth_anything_v2_small
```

模型分别使用其提供方的条款。将本地 MCAP 目录设为 `AGENTOS_DATA_ROOT`，其内含四路 fisheye 视频、前后雷达、LIO／IMU 和 `calibration/calib_para.yml` 附件。

```bash
export AGENTOS_DATA_ROOT=/path/to/AgentOS
export CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export SOM_PYTHON="$PWD/som_local/.venv/bin/python"
export SAM2_CHECKPOINT="$PWD/agentos_dynamic/models/sam2.1_hiera_tiny.pt"
"$SOM_PYTHON" agentos_dynamic/inventory.py --source "$AGENTOS_DATA_ROOT"
"$SOM_PYTHON" agentos_dynamic/read_sensors.py
```

## 基础片段

已有双人交互片段的文件名是 `unified_20260817_160810_0002.mcap`。使用该文件时：

```bash
bash agentos_seated/run.sh
```

`run.sh` 从 15.95 秒开始抽取 8 秒，去畸变输出 768×512，生成掩码、文件袋 SAM2 跟踪、雷达对齐深度、Bootstapir 图像轨迹，然后训练并导出。

换用其他片段时，按下面命令指定文件名和时间窗口，并在 `agentos_seated/track_prop.py` 中调整 `seed`、`box`、`points` 到目标物体；参数依赖所选画面。

```bash
"$SOM_PYTHON" agentos_dynamic/extract_clip.py \
  --root agentos_seated --source-name YOUR_MCAP_STEM \
  --start 15.95 --duration 8 --pitch 0 --fov 95 --width 768 --height 512
"$SOM_PYTHON" agentos_dynamic/export_mcap.py --root agentos_seated
"$SOM_PYTHON" agentos_dynamic/preprocess.py masks --root agentos_seated --include-seated-chairs
"$SOM_PYTHON" agentos_seated/track_prop.py
"$SOM_PYTHON" agentos_dynamic/preprocess.py depth --root agentos_seated
"$SOM_PYTHON" agentos_dynamic/preprocess.py tracks --root agentos_seated
```

办公室站立片段可运行 `bash agentos_dynamic/run.sh`；其文件名、裁剪时间和视域默认值见 `extract_clip.py`。

## V2：掩码、相机和深度优化

先生成上述基础片段与轨迹。

```bash
"$SOM_PYTHON" agentos_seated_v2/prepare_masks.py
"$SOM_PYTHON" agentos_seated_v2/refine_cameras.py
"$SOM_PYTHON" agentos_seated_v2/refine_depth.py
bash agentos_seated_v2/train.sh
"$SOM_PYTHON" agentos_seated_v2/render_results.py
"$SOM_PYTHON" agentos_seated_v2/compare_models.py
```

## V3：逐点雷达深度与辅助相机

先生成基础片段和 V2 处理结果。

```bash
"$SOM_PYTHON" agentos_seated_v3/prepare_sensors.py
"$SOM_PYTHON" agentos_seated_v3/check_reprojection.py
"$SOM_PYTHON" agentos_seated_v3/prepare_inputs.py
"$SOM_PYTHON" agentos_seated_v3/filter_auxiliary.py
bash agentos_seated_v3/train.sh
"$SOM_PYTHON" agentos_seated_v3/render_results.py
"$SOM_PYTHON" agentos_seated_v3/compare_models.py
"$SOM_PYTHON" agentos_seated_v3/evaluate_sensors.py
```

训练会从对应 `runs/office/checkpoints/last.ckpt` 续训。新训练使用新实验目录或将已有 `runs/office` 移到备份目录。输出位于各实验的 `results/office`，包含 RGB、对照视频、PLY、轨迹和指标。

## 统一查看

```bash
python3 scripts/configure_viewer.py --include-private --available-only
bash reconstruction_viewer/start.sh --registry reconstruction_viewer/scenes.local.json
```

打开 <http://127.0.0.1:8890/>。
