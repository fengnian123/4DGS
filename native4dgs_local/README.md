# Yang Native 4DGS 复现

本目录复现 Yang 等人的 ICLR 2024 论文 *Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting*。输入为带相机参数和时间戳的图像，重建结果支持时间控制与自由视角渲染。模型采用一组四维高斯联合表示空间与时间，通过时间条件化获得三维高斯分布，再由 CUDA 光栅器生成图像；四维旋转、时变外观表示和高斯增密均启用。

本方法与仓库已有 `4dgs_local/` 的 Wu 等人 4DGaussians 分属不同论文。本次采用作者发布配置的完整训练预算，覆盖 D-NeRF 两个场景和 Neural 3D Video 一个场景。所有报告指标均来自本次实测；评测范围不包含消融实验和完整数据集平均指标。

内容导航：[结果展示](#结果和查看方式) · [交互查看器](#自由视角实时查看器) · [复现流程](#目录和环境隔离) · [关键技术讲解](#关键技术讲解) · [完整实测报告](REPORT.md)。

## 结果和查看方式

| 场景 | 数据集 | 训练终点 | 最佳测试检查点 | 最佳 PSNR | 终点 PSNR | 分辨率 |
|---|---|---:|---:|---:|---:|---|
| Jumping Jacks | D-NeRF | 30,000 | 4,000 | 33.514 | 30.666 | 400×400 |
| Bouncing Balls | D-NeRF | 20,000 | 18,000 | 33.794 | 33.712 | 400×400 |
| Coffee Martini | Neural 3D Video | 30,000 | 28,000 | 28.317 | 28.243 | 1352×1014 |

在相同像素值范围下，PSNR 越高表示重建图像与参考图像之间的均方误差越低，宜在相同场景和评测设置下比较；细节恢复、时间闪烁和漂浮伪影需结合视频进行定性评估。“最佳”检查点依据同一测试集的 PSNR 选择，不构成独立验证集选模后的盲测。Jumping Jacks 的最佳检查点位于 4,000 步，实际训练仍完成 30,000 步，后期质量下降一并报告。

直接点击下表即可查看文件，无需先安装训练环境。视频链接若被浏览器下载，可本地播放，或使用下面的媒体页面命令。

| 场景 | 预录视频（最佳模型） | 真实图 / 重建 / 误差对比 |
|---|---|---|
| Jumping Jacks | [固定视角动态渲染](results/jumpingjacks_auditfix_full/fixed_view_time.mp4) · [固定时刻视角漫游](results/jumpingjacks_auditfix_full/fixed_time_view.mp4) | [最佳](results/jumpingjacks_auditfix_full/best/test_comparison.png) · [训练终点](results/jumpingjacks_auditfix_full/final/test_comparison.png) |
| Bouncing Balls | [固定视角动态渲染](results/bouncingballs_auditfix_full/fixed_view_time.mp4) · [固定时刻视角漫游](results/bouncingballs_auditfix_full/fixed_time_view.mp4) | [最佳](results/bouncingballs_auditfix_full/best/test_comparison.png) · [训练终点](results/bouncingballs_auditfix_full/final/test_comparison.png) |
| Coffee Martini | [固定视角动态渲染](results/coffee_martini_auditfix_full/fixed_view_time.mp4) · [固定时刻视角漫游](results/coffee_martini_auditfix_full/fixed_time_view.mp4) | [最佳](results/coffee_martini_auditfix_full/best/test_comparison.png) · [训练终点](results/coffee_martini_auditfix_full/final/test_comparison.png) |

每个 `results/<场景>/` 包含：

- `metrics.json`：最佳、终点、选用版本以及所有测试帧的指标；保留完整精度。
- `best/test_comparison.png` 与 `final/test_comparison.png`：每行依次为真实图、渲染图、绝对误差放大 5 倍；三行为首帧、中间帧、末帧。
- `fixed_view_time.mp4`：固定相机、时间变化的模型渲染。
- `fixed_time_view.mp4`：固定时间、相机转动的模型渲染。

两类视频均使用最佳检查点，150 帧、30 fps，共 5 秒。Coffee 的约 10 秒动态被压到 5 秒展示。视频没有逐帧真实画面对照，也不能自由拖动视角。自由视角需要下面的实时查看器和模型权重。

在合入本仓库后，不需要 GPU 即可观看视频：

```bash
python3 reconstruction_viewer/gallery.py --registry reconstruction_viewer/scenes.native4dgs.json
```

打开 `http://127.0.0.1:8900/`。完整指标和边界条件见 [REPORT.md](REPORT.md)。

## 目录和环境隔离

从仓库根目录执行以下命令。`NATIVE4DGS_ROOT` 指向专属实验目录；环境、数据、缓存、日志、权重都在这个目录内。代码和配置模板由本仓库管理。示例中的 `/path/to/my_native4dgs` 需要换成自己的实验目录。

```bash
export NATIVE4DGS_ROOT=/path/to/my_native4dgs
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=8.6
export CUDA_VISIBLE_DEVICES=0
export MAX_JOBS=4 OMP_NUM_THREADS=4
bash native4dgs_local/setup.sh
export PATH="$NATIVE4DGS_ROOT/envs/train/bin:$CUDA_HOME/bin:$PATH"
```

记录的环境为 Linux、Python 3.10.14、PyTorch 2.7.1+cu128、CUDA 12.8、RTX A6000 48 GB。`8.6` 是本次 GPU 架构，不是所有 GPU 的通用值。需要系统已有 Git、CUDA 编译器和 FFmpeg；无需改动共享系统环境。完整依赖记录在 `environment/`。安装脚本使用精简运行依赖，完整 freeze 用于溯源。

`bootstrap.py` 获取固定提交 `121e360ce23d6e9ca7734072393414c844dfaea2`，应用 `patches/native4dgs_yang2024.patch`，然后核对 766 个源码文件的 SHA-256。已有代码目录只核验，不覆盖。补丁包含本次实际使用的数值修正和兼容修改，说明见报告。

## 数据准备

数据来源、下载地址和原始压缩包校验值见 [DATA.md](DATA.md)。原始数据不放进 Git。

D-NeRF 解压后，向脚本传入含 `jumpingjacks/`、`bouncingballs/` 的目录。默认建立链接，避免复制原始图片；不沿用来源目录中可能存在的旧初始化点云。

```bash
python native4dgs_local/prepare_dnerf.py --source /path/to/extracted/data
```

Coffee Martini 将 18 个 `cam*.mp4` 和 `poses_bounds.npy` 放在 `$NATIVE4DGS_ROOT/data/N3V_raw/coffee_martini/`。预处理使用独立环境，生成全量帧、相机文件和稠密初始化点云：

```bash
python3.10 -m venv "$NATIVE4DGS_ROOT/envs/preprocess"
"$NATIVE4DGS_ROOT/envs/preprocess/bin/python" -m pip install -r native4dgs_local/environment/preprocess-freeze.txt
"$NATIVE4DGS_ROOT/envs/preprocess/bin/python" native4dgs_local/prepare_n3v.py
```

PyCOLMAP 需能使用 CUDA，`assert pycolmap.has_cuda` 会先检查。该环境的 CUDA 运行库版本为 12.9，训练环境仍是 12.8；不要混装这两个环境。相机来自官方 `poses_bounds.npy`，初始化只用 17 个训练相机的第 0 帧，不使用留出的 `cam00`。

## 完整训练与评测

先检查解析后的配置，再训练一个场景。`--dry-run` 只显示配置。

```bash
python native4dgs_local/train.py jumpingjacks_auditfix_full --dry-run
python native4dgs_local/train.py jumpingjacks_auditfix_full
python native4dgs_local/evaluate.py jumpingjacks_auditfix_full

python native4dgs_local/train.py bouncingballs_auditfix_full
python native4dgs_local/evaluate.py bouncingballs_auditfix_full

python native4dgs_local/train.py coffee_martini_auditfix_full
python native4dgs_local/evaluate.py coffee_martini_auditfix_full
```

三个命令分别保留 30,000、20,000、30,000 步预算。Coffee 自动启用 CPU 暂存 Adam 状态，仍使用完整分辨率、批大小及增密计划；它需要充足主存，且 CPU/GPU 数据搬运会增加训练时间。GPU 由 `CUDA_VISIBLE_DEVICES` 显式选择，脚本不调度或停止其他人的任务。

训练日志在 `logs/`，模型在 `runs/`，新评测在实验根目录的 `results/`。训练不会覆盖已有检查点，评测不会自动删除旧权重。续训需显式提供 `--resume <路径> --best-psnr <历史最佳值>`；Coffee 只能从环境图冻结之后的检查点继续。发布版检查点不含随机数及数据加载器状态，续训采样顺序不能逐位恢复。

## 自由视角实时查看器

具备模型和数据后，运行本方法的独立界面：

```bash
python native4dgs_local/viewer.py --port 8765
```

打开 `http://127.0.0.1:8765/`，自动播放时仍可拖动旋转，右键平移、滚轮缩放；也可切换最佳/终点、参考相机、颜色/深度/透明度显示。远程运行时，通过自己的 SSH 连接转发该回环端口。仓库不包含任何个人登录信息。

接入仓库统一查看器时，复制 `reconstruction_viewer/scenes.native4dgs.json` 为本机配置，将 `root`、`python`、`code_root` 改为实验根目录、该环境的 Python 和修正源码目录，然后：

```bash
bash reconstruction_viewer/start.sh --registry reconstruction_viewer/scenes.native4dgs.local.json
```

统一界面沿用原仓库的 Viser 依赖。它有六个条目，分别对应三个场景的最佳和终点。指标对应当前条目；预录视频仅放在最佳条目中。适配器实现既有 `metadata` / `render(request)` 接口。

## 权重与清理

权重索引见 [models.json](models.json)。本仓库不包含权重；已有六份权重共约 32.7 GB，尚未设置公共下载地址。`viewer_model.pth` 去除了优化器状态，适合浏览；`chkpnt<终点>.pth` 保留终点训练状态。没有权重和相机数据时只能查看预录结果。

停止属于自己的查看器和训练进程后，可清理自己配置的实验根目录以移除环境、缓存、数据和权重。先确认输出已备份、路径确属本实验；这里不提供自动递归删除脚本。使用链接导入 D-NeRF 时，实验目录中的链接和外部原始数据是分开的。

## 论文与许可

[官方源码](https://github.com/LogosIntelligenceLab/4d-gaussian-splatting) · [论文页面](https://logosintelligencelab.github.io/4d-gaussian-splatting/)

上游主体为 MIT；继承的 Gaussian Splatting 组件保留其研究和非商业使用条款。Coffee 媒体保留 Neural 3D Video 的 CC BY-NC 4.0 条件。数据、模型和子组件不能一律按软件 MIT 处理；完整说明见仓库 `THIRD_PARTY.md` 和 `licenses/`。

```bibtex
@inproceedings{yang2023gs4d,
  title={Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting},
  author={Yang, Zeyu and Yang, Hongye and Pan, Zijie and Zhang, Li},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## 关键技术讲解

本方法采用一组原生四维高斯联合建模空间结构与时间变化。给定时间后，由条件高斯分布得到三维均值与协方差，并结合时间边缘权重进行渲染。对单个高斯，条件均值随时间线性变化，条件协方差保持不变；多个具有不同时间权重和运动参数的高斯共同表示场景中的非线性运动与非刚性形变。相关推导见 [论文第 3.2 节](https://arxiv.org/html/2310.10642v3#S3.SS2)。

| 关键技术 | 作用 | 实现机制 |
|---|---|---|
| **四维旋转与时空耦合** | 联合建模空间结构与时间变化，支持跨时间参数共享 | 通过四维旋转与尺度参数构造时空协方差，利用时间条件化计算三维均值与协方差，无需额外的形变网络。 |
| **时间边缘权重** | 控制高斯的有效时间范围 | 将时间边缘高斯权重乘入不透明度；局部动态可由较窄的时间分布表示，静态区域可由较宽的时间分布持续表示。 |
| **四维外观基函数（4DSH）** | 刻画视角相关且随时间变化的外观 | 组合空间球谐基与时间余弦基，拟合颜色和反射外观随视角、时间的变化；该表示不显式分解材质与光照。 |
| **时空增密与剪枝** | 提高空间细节与快速时间变化的表示能力 | 依据空间位置与时间中心的梯度进行增密和分裂，同时剪除贡献较低的高斯，实现自适应密度控制。 |
| **跨时间批量采样** | 缓解时间闪烁与抖动 | 在训练批次中采样不同时刻的图像，联合优化共享高斯参数，降低单帧采样引起的更新波动。 |
| **可微高斯泼溅渲染** | 兼顾渲染效率与端到端优化 | 将条件三维高斯投影至二维，按深度执行透明度合成，并通过渲染误差反向传播优化几何、颜色与不透明度参数。 |

上表归纳自 [论文第 3.2–3.3 节](https://arxiv.org/html/2310.10642v3#S3.SS2) 及 [附录 E 的时间特性分析](https://arxiv.org/html/2310.10642v3#A5)。四维表示、配套渲染与 4DSH 是论文主要贡献；可微渲染和密度控制沿用并扩展了 3DGS 的基础，批量采样属于训练设计。

本实现通过优化场景几何、外观及时间分布提高重建质量，未引入额外的图像修复或后处理增强模块。训练结束后，高斯总数固定；各时刻的空间均值、颜色、贡献权重及筛选后参与渲染的高斯数量随时间变化。本次评测针对观测时间范围内的动态重建与新视角合成，未评估未来运动外推能力。

本次复现还包含梯度与透明度加载修正，以及完整训练预算下的显存管理。这些属于复现中的数值与工程改进，具体修改及验证证据见 [REPORT.md](REPORT.md#相对官方发布代码的修改)。
