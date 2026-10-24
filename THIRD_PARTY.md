# Upstream sources and attribution

Training implementations are downloaded by `scripts/bootstrap.py`; each patch is against the pinned commit below. Upstream copyright notices and licenses remain in those checkouts.

| Component | Source | Commit / license |
|---|---|---|
| 4D Gaussian Splatting | https://github.com/hustvl/4DGaussians | `843d5ac636c37e4b611242287754f3d4ed150144`, Apache-2.0; CUDA submodules retain their separate licenses |
| Shape of Motion | https://github.com/vye16/shape-of-motion | `579753e1c7ba96f60cd7690e5b835627bd1935e9`, MIT |
| simple-knn | https://gitlab.inria.fr/bkerbl/simple-knn | `44f764299fa305faf6ec5ebd99939e0508331503`, upstream license applies |
| depth-diff-gaussian-rasterization | https://github.com/ingra14m/depth-diff-gaussian-rasterization | `e49506654e8e11ed8a62d22bcb693e943fdecacf`, upstream license applies |

`patches/som.patch` contains compatibility fixes for the public reproduction. `som-agentos-v2.patch` and `som-agentos-v3.patch` are each complete, independent diffs against the same upstream commit, not patches to apply on top of one another. They add camera/depth/mask handling and sensor constraints to the local experiments. `fourdgs.patch` and `simple-knn.patch` contain loader, finite-loss and current-toolchain compatibility changes. See `licenses/` for the parent implementation licenses.

## Data and rendered examples

- **Coffee Martini**: [Neural 3D Video Synthesis](https://github.com/facebookresearch/Neural_3D_Video), Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, Zhaoyang Lv, CVPR 2022. The dataset is [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Files under `4dgs_local/results/coffee_7000/` are locally generated reconstructions, resized reference images, labeled comparisons, virtual-camera videos, and numerical evaluations derived from that dataset. These media retain the noncommercial attribution terms, not a blanket software license. License text: `licenses/Neural-3D-Video-CC-BY-NC-4.0.txt`.
- **3D Printer**: [HyperNeRF](https://github.com/google/hypernerf), Keunhong Park et al., ACM TOG 2021; data from the [official release](https://github.com/google/hypernerf/releases/tag/v0.1), distributed with the project's Apache-2.0 license. Result images/videos are local reconstructions and labeled comparisons. The calibrated variant undistorts/reprojects the source images and changes the field of view; its deformation learning rate is reduced by 10×. Original data and the upstream-provided initialization cloud are obtained separately. License text: `licenses/HyperNeRF-Apache-2.0.txt`.
- **Balloon1**: Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, Jan Kautz, *Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera*, CVPR 2020. [Dataset terms](https://gorokee.github.io/jsyoon/dynamic_synth/) limit use to research and prohibit redistribution of original or modified data without author permission. This repository includes only scripts, provider links and numerical evaluations, not Balloon1 imagery, videos, reconstructed geometry or trained weights. Obtain authorized inputs directly from the providers before running the pipeline locally.
- **AgentOS**: private sensor recordings are not distributed. Scripts and numerical summaries are included; MCAPs, calibration/identity attachments, camera frames, derived videos, tracks and model weights remain local.

Software licenses do not override dataset, model-weight, privacy or submodule terms. This repository does not grant a new blanket license over third-party material.
