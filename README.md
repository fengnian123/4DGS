# 4DGS 动态重建与统一查看器

本仓库收录多种动态重建方法。Yang 原生 4DGS 的三场景复现可从 [结果与运行说明](native4dgs_local/README.md) 开始，或直接阅读 [关键技术的通俗理解](native4dgs_local/README.md#关键技术的通俗理解)。

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

## Yang Native 4DGS 复现

`native4dgs_local/` 对应 Yang 等人的 ICLR 2024 原生四维高斯方法，与 `4dgs_local/` 的 Wu 4DGaussians 分开。输入带相机参数和时间戳的图像，训练后得到可以调时间、换视角的动态三维场景。

本次完成两个数据集的三个场景，保留作者发布配置的完整训练步数。下表均为本次实测；相同场景和评测设置下，PSNR 越高越好。“最佳”按同一测试集选择，需与终点一起看。

| 场景 | 数据集 | 完成步数 | PSNR：最佳 / 终点 | 直接看效果 |
|---|---|---:|---:|---|
| Jumping Jacks | D-NeRF | 30,000 | 33.514 / 30.666 | [动态视频](native4dgs_local/results/jumpingjacks_auditfix_full/fixed_view_time.mp4) · [最佳对比图](native4dgs_local/results/jumpingjacks_auditfix_full/best/test_comparison.png) · [终点对比图](native4dgs_local/results/jumpingjacks_auditfix_full/final/test_comparison.png) |
| Bouncing Balls | D-NeRF | 20,000 | 33.794 / 33.712 | [动态视频](native4dgs_local/results/bouncingballs_auditfix_full/fixed_view_time.mp4) · [最佳对比图](native4dgs_local/results/bouncingballs_auditfix_full/best/test_comparison.png) · [终点对比图](native4dgs_local/results/bouncingballs_auditfix_full/final/test_comparison.png) |
| Coffee Martini | Neural 3D Video | 30,000 | 28.317 / 28.243 | [动态视频](native4dgs_local/results/coffee_martini_auditfix_full/fixed_view_time.mp4) · [最佳对比图](native4dgs_local/results/coffee_martini_auditfix_full/best/test_comparison.png) · [终点对比图](native4dgs_local/results/coffee_martini_auditfix_full/final/test_comparison.png) |

对比图每行依次为真实图、重建图、误差放大 5 倍。视频使用最佳检查点；Jumping Jacks 最佳为 4,000 步，但已继续训练到 30,000 步，后期质量下降如实保留。更多视角视频、分辨率和指标说明见 [方法 README](native4dgs_local/README.md#结果和查看方式)。

仅观看结果无需 GPU 或模型权重，从仓库根目录运行：

```bash
python3 reconstruction_viewer/gallery.py --registry reconstruction_viewer/scenes.native4dgs.json
```

打开 <http://127.0.0.1:8900/> 播放预录视频。希望一边播放一边拖动视角，见 [实时查看器说明](native4dgs_local/README.md#自由视角实时查看器)；需要本机模型、相机数据和 CUDA 环境。权重未放入 Git，公共下载地址尚未设置，详情见 [权重索引](native4dgs_local/models.json)。

**方法怎么理解？** 可以把它想成许多在局部时间段内匀速移动、逐渐显隐的小高斯，共同拼出运动和形变。四维旋转关联空间与时间，时间权重控制显隐，4D 颜色基描述随视角和时间变化的外观，时空增密补足难以拟合的细节。单颗高斯的线性运动不限制整个物体只能匀速运动；这些技术改善的是场景表示与拟合，不属于图像修复。详见 [关键技术的通俗理解](native4dgs_local/README.md#关键技术的通俗理解) 和 [原论文第 3 节](https://arxiv.org/html/2310.10642v3#S3)。

复现代码由固定上游版本加完整补丁恢复。我们的梯度、透明度加载及显存管理修正另见 [实测报告](native4dgs_local/REPORT.md)，与论文提出的方法区分记录；本次未做完整数据集平均分或消融实验。
