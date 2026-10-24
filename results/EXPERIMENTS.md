# Recorded evaluations

All numbers below are measured outputs of the local runs, not copied paper scores. Dataset-specific masks, image sizes, camera coverage and stopping points differ; do not rank methods across these rows.

| Run | Evaluated images | PSNR | SSIM | Evaluation scope |
|---|---:|---:|---:|---|
| 4DGS Coffee Martini, fine 7,000 | 300 | 27.4300 | 0.9015 | held-out camera, full RGB |
| 4DGS 3D Printer, fine 14,000 | 207 | 22.0197 | 0.7062 | public validation split, full RGB |
| 4DGS calibrated printer, fine 14,000 | 207 | 19.5511 | 0.7161 | changed field of view and 0.1× deformation learning rate |
| Shape of Motion Balloon1, 12,000 steps | 576 | 23.9763 | 0.7987 | author co-visible masks; LPIPS 0.05666 |
| AgentOS seated V1, 5,000 steps | 80 | 26.0993 | 0.8679 | training-view reconstruction |
| AgentOS seated V2, 8,010 steps | 80 | 29.6771 | 0.9287 | training-view reconstruction |
| AgentOS seated V3, 6,020 steps | 80 | 29.2304 | 0.9200 | training-view reconstruction with LiDAR / auxiliary-camera constraints |

Coffee uses 676×507 images, 3,000 coarse iterations and the 7,000-fine-iteration snapshot from an unchanged 14,000-step schedule. The local run was manually stopped after inspecting sparse validation frames; this is not a blind benchmark or exact reproduction of the paper's table. `run_scene.py coffee` normally continues to the full schedule. The initialization uses only training cameras at frame 0. The printer uses an upstream initialization cloud whose generation provenance was not independently audited. Its calibrated version changes both the observed pixels and learning rates, not a controlled camera-only ablation.

Balloon1 uses author-preprocessed camera, depth and tracking inputs, 192 training images, 576 validation images and 500 epochs. Full-RGB mean PSNR is 23.0881 dB. Masked SSIM / LPIPS follow the upstream validator and should not be interpreted as full-frame metrics.

AgentOS V3 was freshly initialized, not warm-started from V2. Withheld per-return LiDAR IDs are excluded consistently across views from completion and training losses. Foreground depth median discrepancy to those held-out returns changes from 0.16148 m (V2) to 0.00698 m (V3); file-bag discrepancy changes from 0.11626 m to 0.00631 m. This measures consistency with the same recording, not independent calibration accuracy. Bag mean discrepancy remains 0.09956 m. The training-view bag PSNR improves from 25.1344 to 26.3823 dB, while full-RGB PSNR declines. The common-mask background foreground-leak proxy rises from 0.00309 to 0.00919, and some novel-view streaking worsens. V3 is therefore not an overall artifact-removal improvement or a validated navigation environment. Physical exposure offset, human scan motion and upstream deskew are not independently established.

Per-frame numerical results are preserved in the adjacent JSONs and each public method's `results/` directory. Model checkpoints and original sensor/dataset files are acquired or trained locally through the documented commands. Saved videos show rendered model outputs and labeled reference images; the interactive viewer separately renders models at the requested camera and time.
