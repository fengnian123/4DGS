# 4DGS 动态重建与统一查看器

## 查看已有结果

```bash
git clone https://github.com/fengnian123/4DGS.git
cd 4DGS
python3 reconstruction_viewer/gallery.py
```

打开 <http://127.0.0.1:8900/> 查看厨房和打印机的真实／重建对照视频，以及各实验的逐帧指标。

## 安装训练环境

使用 Linux、Python 3.11、NVIDIA GPU、CUDA 12.8 编译工具链和 FFmpeg。按本机 GPU 设置 `TORCH_CUDA_ARCH_LIST`；下面的 `8.9` 对应 Ada GPU。

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=4 OMP_NUM_THREADS=4
python3 scripts/bootstrap.py som 4dgs
bash scripts/setup.sh som
bash scripts/setup.sh 4dgs
```

`bootstrap.py` 下载固定版本上游代码并应用 `patches/` 中对应补丁。

## 复现 4DGS

```bash
python3 scripts/download_public.py coffee
python3 scripts/download_public.py printer
```

打印机初始化点云从 [4DGS 作者数据](https://drive.google.com/file/d/1fUHiSgimVjVQZ2OOzTFtz02E9EqCoWr5/view)下载，解压到 `4dgs_local/data/author_clouds/`，形成 `hypernerf/virg/virg-3dprinter/points3D_downsample2.ply`。

```bash
cd 4dgs_local
.venv/bin/python prepare_data.py coffee
.venv/bin/python prepare_frames.py
.venv/bin/python prepare_data.py printer
.venv/bin/python run_scene.py coffee
.venv/bin/python run_scene.py printer
.venv/bin/python calibrate_printer.py
.venv/bin/python run_scene.py printer_calibrated
cd ..
```

使用已有模型重新渲染；`--iteration` 指定保存的 fine 阶段模型。

```bash
cd 4dgs_local
.venv/bin/python run_scene.py coffee --eval-only --iteration 7000
.venv/bin/python run_scene.py printer --eval-only --iteration 14000
.venv/bin/python run_scene.py printer_calibrated --eval-only --iteration 14000
cd ..
```

## 复现 Shape of Motion

先阅读 [Balloon1 数据使用条款](https://gorokee.github.io/jsyoon/dynamic_synth/)，从[作者预处理数据](https://drive.google.com/drive/folders/1xzn-Mu_jyr-JTsrERRU-Mh2hQ-NWdfv8)下载。

```bash
cd som_local
.venv/bin/python download_data.py --id 1atIXDSiNRCh7kOaoB4xU2UlerDZ3CouX --name Balloon1
.venv/bin/python prepare_data.py
bash train.sh
.venv/bin/python evaluate.py --output results/balloon1
cd ..
```

## 启动交互查看器

将 `reconstruction_viewer/scenes.all.example.json` 中的 `iteration` 设为要浏览的保存步数，或注册已有本地模型目录。模型保存在各方法的 `runs/`，数据保存在 `data/`。

```bash
som_local/.venv/bin/python -m pip install -r requirements-viewer.txt
python3 scripts/configure_viewer.py --available-only
bash reconstruction_viewer/start.sh --registry reconstruction_viewer/scenes.local.json
```

打开 <http://127.0.0.1:8890/>。选择场景，拖动旋转、右键平移、滚轮缩放；时间滑块和播放按钮控制动态状态。“恢复参考视角”复位相机，“保存当前画面”导出渲染。媒体页面位于 <http://127.0.0.1:8900/>。切换到实时查看器前，先停止单独运行的 `gallery.py`，避免端口冲突。

注册另一目录下的已有结果：

```bash
python3 scripts/configure_viewer.py \
  --projects-root /path/to/projects \
  --experiments-root /path/to/experiments \
  --include-private --available-only
bash reconstruction_viewer/start.sh --registry reconstruction_viewer/scenes.local.json
```

目录下分别放置 `4dgs_local`、`som_local` 和 `agentos_*` 实验；查看器按每个场景配置的 Python 环境启动渲染进程。新方法在 `reconstruction_viewer/adapters/` 实现适配器，注册格式见 `scenes.all.example.json`。`metadata` 提供 OpenCV `c2w`、垂直 `fov`、`resolution`、`times`、`initial_frame`、`look_at`、`gaussians`；`render(request)` 接收 `w2c`、`fov`、`size`、`time` 并返回 `uint8 RGB`。

## 自采数据处理

按 [AgentOS 处理命令](agentos_dynamic/README.md)运行去畸变、掩码、深度、轨迹、相机优化和实测雷达约束训练。

## 测试

```bash
python3 -m unittest discover -s tests -v
som_local/.venv/bin/python reconstruction_viewer/check_renderers.py --registry reconstruction_viewer/scenes.local.json
som_local/.venv/bin/python -m pip install playwright
som_local/.venv/bin/python -m playwright install chromium
som_local/.venv/bin/python reconstruction_viewer/check_ui.py --registry reconstruction_viewer/scenes.local.json
```
