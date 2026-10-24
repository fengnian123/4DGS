import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import least_squares
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'
INCLUDE_SEATED_CHAIRS = False
SEQUENCE = ROOT / "agentos_office"
SOM = ROOT.parent / "som_local/shape-of-motion"
sys.path.insert(0, str(SOM / "preproc"))
torch.set_float32_matmul_precision("high")


def prepare():
    (SEQUENCE / "images").mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "clip/front").glob("[0-9]*.png")):
        shutil.copyfile(path, SEQUENCE / "images" / path.name)
    return sorted((SEQUENCE / "images").glob("*.png"))


@torch.no_grad()
def masks(files):
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, min_size=480, max_size=640).cuda().eval()
    destination = SEQUENCE / "masks"
    destination.mkdir(exist_ok=True)
    validation = []
    for index, path in enumerate(files):
        image = np.asarray(Image.open(path).convert("RGB"))
        prediction = model([torch.from_numpy(image.copy()).permute(2, 0, 1).float().cuda() / 255])[0]
        people = (prediction["labels"] == 1) & (prediction["scores"] >= .65)
        selected = people.clone()
        if INCLUDE_SEATED_CHAIRS and people.any():
            person_boxes = prediction['boxes'][people]
            for chair in torch.where((prediction['labels'] == 62) & (prediction['scores'] >= .6))[0]:
                box = prediction['boxes'][chair]
                intersection = (torch.minimum(person_boxes[:, 2:], box[2:]) - torch.maximum(person_boxes[:, :2], box[:2])).clamp(min=0).prod(-1)
                overlap = intersection / torch.minimum((person_boxes[:, 2:] - person_boxes[:, :2]).prod(-1), (box[2:] - box[:2]).prod()).clamp(min=1)
                if overlap.max() > .15:
                    selected[chair] = True
        mask = (prediction["masks"][selected, 0] > .5).any(0).cpu().numpy().astype(np.uint8)
        cv2.imwrite(str(destination / path.name), mask * 255)
        overlay = image.copy()
        overlay[mask > 0] = (overlay[mask > 0] * .6 + np.array([0, 190, 80]) * .4).astype(np.uint8)
        if index in (0, len(files) // 2, len(files) - 1):
            Image.fromarray(overlay).save(destination / f"preview_{index:05d}.jpg")
        validation.append({"frame": path.stem, "person_instances": int(people.sum()), "chair_instances": int((selected & (prediction['labels'] == 62)).sum()), "foreground_fraction": float(mask.mean())})
        if index % 10 == 0:
            print("Masks", index, validation[-1], flush=True)
    (ROOT / "mask_quality.json").write_text(json.dumps(validation, indent=2))


@torch.no_grad()
def depth(files):
    model_dir = MODELS / "depth_anything_v2_small"
    processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(model_dir, local_files_only=True).cuda().eval()
    depths, diagnostics = [], []
    rng = np.random.default_rng(42)
    for index, path in enumerate(files):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        prediction = model(**inputs).predicted_depth
        prediction = torch.nn.functional.interpolate(prediction[:, None], size=image.size[::-1], mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        low, high = np.quantile(prediction, [.01, .99])
        mono = (prediction - low) / max(high - low, 1e-6)
        anchors = np.load(ROOT / "clip/lidar_depth" / f"{path.stem}.npy")
        valid = (anchors > .4) & (anchors < 15) & (mono > -.05) & (mono < 1.05)
        x, y = mono[valid], 1 / anchors[valid]
        chosen = rng.choice(len(x), min(6000, len(x)), replace=False)
        split = max(10, int(len(chosen) * .8))
        train, test = chosen[:split], chosen[split:]
        fit = least_squares(lambda parameters: parameters[0] * x[train] + parameters[1] - y[train], [.7, .2], bounds=([1e-4, 1e-3], [10, 10]), loss="soft_l1", f_scale=.025)
        disparity = np.maximum(fit.x[0] * mono + fit.x[1], 1 / 15)
        array = np.clip(1 / disparity, .3, 15).astype(np.float32)
        depths.append(array)
        residual = float(np.median(np.abs(1 / np.maximum(fit.x[0] * x[test] + fit.x[1], 1 / 15) - 1 / y[test])))
        diagnostics.append({"frame": path.stem, "lidar_anchor_count": int(valid.sum()), "inverse_depth_scale": float(fit.x[0]), "inverse_depth_shift": float(fit.x[1]), "heldout_anchor_median_error_m": residual})
        if index in (0, len(files) // 2, len(files) - 1):
            color = cv2.applyColorMap((np.clip(array / 5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(str(ROOT / f"depth_preview_{index:05d}.jpg"), color)
        if index % 10 == 0:
            print("Depth", index, diagnostics[-1], flush=True)
    selection = json.loads((ROOT / "clip/selection.json").read_text())
    np.savez_compressed(SEQUENCE / "agentos_office.npz", cam_c2w=np.load(ROOT / "clip/front/poses.npy"), intrinsic=np.asarray(selection["intrinsic"], dtype=np.float32), depths=np.asarray(depths))
    (ROOT / "depth_quality.json").write_text(json.dumps(diagnostics, indent=2))


@torch.no_grad()
def tracks(files):
    from tapnet_torch.tapir_model import TAPIR
    model = TAPIR(pyramid_level=1)
    model.load_state_dict(torch.load(MODELS / "bootstapir_checkpoint_v2.pt", map_location="cpu", weights_only=True))
    model = model.cuda().eval()
    images = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in files])
    count, height, width = images.shape[:3]
    resized = np.stack([cv2.resize(image, (256, 256)) for image in images])
    video = torch.from_numpy(resized).float().cuda()[None] / 127.5 - 1
    features = model.get_feature_grids(video, is_training=False)
    destination = SEQUENCE / "bootstapir"
    destination.mkdir(exist_ok=True)
    ys, xs = np.mgrid[0:height:10, 0:width:10]
    for source, path in enumerate(files):
        if len(list(destination.glob(f"{path.stem}_*.npy"))) == count:
            continue
        mask = np.asarray(Image.open(SEQUENCE / "masks" / path.name))
        valid = mask[ys, xs] > 0
        points_original = np.stack([xs[valid], ys[valid]], axis=-1)
        query = np.stack([np.full(valid.sum(), source), ys[valid] / (height - 1) * 255, xs[valid] / (width - 1) * 255], axis=-1).astype(np.float32)
        outputs = []
        for chunk in np.array_split(query, max(1, int(np.ceil(len(query) / 128)))):
            if len(chunk) == 0:
                continue
            points = torch.from_numpy(chunk)[None].cuda()
            descriptors = model.get_query_features(video, False, points, features)
            prediction = model.estimate_trajectories((256, 256), False, features, descriptors, points, query_chunk_size=64)
            p = model.num_pips_iter
            values = {key: torch.stack(prediction[key][p::p]).mean(0)[0].cpu().numpy() for key in ("tracks", "occlusion", "expected_dist")}
            values["tracks"][..., 0] *= (width - 1) / 255
            values["tracks"][..., 1] *= (height - 1) / 255
            outputs.append(np.concatenate([values["tracks"], values["occlusion"][..., None], values["expected_dist"][..., None]], axis=-1))
        array = np.concatenate(outputs) if outputs else np.zeros((0, count, 4), np.float32)
        array[:, source, :2] = points_original
        for target, other in enumerate(files):
            np.save(destination / f"{path.stem}_{other.stem}.npy", array[:, target])
        print(f"Tracks {source + 1}/{count}, {len(array)} foreground queries", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["masks", "depth", "tracks", "all"])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--models-root', type=Path, default=MODELS)
    parser.add_argument('--include-seated-chairs', action='store_true')
    args = parser.parse_args()
    ROOT, MODELS = args.root.resolve(), args.models_root.resolve()
    INCLUDE_SEATED_CHAIRS = args.include_seated_chairs
    SEQUENCE = ROOT / 'agentos_office'
    files = prepare()
    for stage in ("masks", "depth", "tracks"):
        if args.stage in (stage, "all"):
            globals()[stage](files)
