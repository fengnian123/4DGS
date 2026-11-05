# 数据来源与准备范围

## D-NeRF

作者：Albert Pumarola、Enric Corona、Gerard Pons-Moll、Francesc Moreno-Noguer。论文 *D-NeRF: Neural Radiance Fields for Dynamic Scenes*，CVPR 2021。

[官方仓库与下载入口](https://github.com/albertpumarola/D-NeRF#download-dataset) · [项目页面](https://www.albertpumarola.com/research/D-NeRF/index.html)

本次使用公开 D-NeRF 数据的 ModelScope 镜像 `Damo_XR_Lab/D_NeRF_Dataset`，记录版本 `2ed5d4d1202c0e7dfc1bc2794ea86c953e58fe49`，只使用 `jumpingjacks` 和 `bouncingballs`。下载压缩包 SHA-256：

```text
4e901bd3463e874bed5dd6bbe3236a8b2348f9adb5ef20484f8ba8b6d96876ec
```

官方镜像或重新打包后的压缩包可能字节不同，不能仅凭文件名认为同一份数据。`prepare_dnerf.py` 保留各场景的原始 train/test/val 布局，训练采用配置中的缩放系数 2；未打包原始数据。

结果媒体是本次模型重建和少量真实帧的研究对照，注明 D-NeRF 来源。`licenses/D-NeRF-code-MIT.txt` 是上游**代码**许可副本，不据此将所有数据和衍生媒体声明为 MIT。

## Neural 3D Video 的 Coffee Martini

数据：Tianye Li 等，*Neural 3D Video Synthesis from Multi-view Video*，CVPR 2022。

[官方项目](https://github.com/facebookresearch/Neural_3D_Video) · [v1.0 数据发布](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) · [Coffee Martini 压缩包](https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip)

本次原始压缩包 SHA-256：

```text
cbc31291ce143e6f31a00f23a049663a61cc8793af8dab259dbc6e338a676816
```

将视频和 `poses_bounds.npy` 放入专属实验目录的 `data/N3V_raw/coffee_martini/`。预处理保留原始 2704×2028 图像，在训练加载时缩放为 1352×1014；抽取每个相机全部 300 帧，时间为帧号除以 30。`cam00` 留出测试，另外 17 个相机训练。

相机转换使用固定上游版本的 `scripts/n3v2blender.py` 中的几何函数。稠密点云由 17 个训练相机第 0 帧的固定相机三角化、几何 PatchMatch 和融合得到，未将测试相机用于初始化。

数据及衍生媒体保留 CC BY-NC 4.0 的署名和非商业条件；许可见 [Neural-3D-Video-CC-BY-NC-4.0.txt](../licenses/Neural-3D-Video-CC-BY-NC-4.0.txt)。这里的媒体是本地重建、对比图及虚拟相机渲染，不重新授权原始数据。
