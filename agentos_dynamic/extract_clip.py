import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=ROOT)
parser.add_argument('--source-name', default='unified_20260817_154735_0000')
parser.add_argument('--start', type=float, default=4.)
parser.add_argument('--duration', type=float, default=8.)
parser.add_argument('--pitch', type=float, default=20.)
parser.add_argument('--fov', type=float, default=115.)
parser.add_argument('--width', type=int, default=640)
parser.add_argument('--height', type=int, default=480)
args = parser.parse_args()
source = next(Path(record["file"]) for record in json.loads((ROOT / "inventory.json").read_text())["files"] if Path(record["file"]).stem == args.source_name)
out = args.root.resolve() / "clip"
out.mkdir(parents=True, exist_ok=True)
start_s, duration_s = args.start, args.duration
cameras = ["front", "left", "right", "rear"]
packets = {name: [] for name in cameras}
odometry, clouds = [], []
start_ns = None
with source.open("rb") as stream:
    reader = make_reader(stream)
    for attachment in reader.iter_attachments():
        if attachment.name == "calibration/calib_para.yml":
            (out / "calib_para.yml").write_bytes(attachment.data)
            calibration = yaml.safe_load(attachment.data.decode().split("\n", 1)[1])
    factory = DecoderFactory()
    topics = [f"/sensor/camera/{name}_fisheye_encoding" for name in cameras] + ["/LIO_ODOM_HIGH_FREQUENCY", "/sensor/lidar/front_lidar_pcd"]
    for schema, channel, message in reader.iter_messages(topics=topics):
        decoded = factory.decoder_for(channel.message_encoding, schema)(message.data)
        if "camera" in channel.topic:
            stamp = decoded.timestamp.sec * 10**9 + decoded.timestamp.nanosec
            if start_ns is None:
                start_ns = stamp + round(start_s * 1e9)
            if start_ns <= stamp < start_ns + duration_s * 1e9:
                name = channel.topic.split("/")[-1].split("_")[0]
                packets[name].append((stamp, message.log_time, bytes(decoded.data)))
        elif "ODOM" in channel.topic:
            pose = decoded.pose.pose
            odometry.append([decoded.header.stamp.sec * 10**9 + decoded.header.stamp.nanosec, pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        else:
            stamp = decoded.header.stamp.sec * 10**9 + decoded.header.stamp.nanosec
            if start_ns is not None and start_ns - 2e8 <= stamp <= start_ns + (duration_s + .2) * 1e9:
                raw = bytes(decoded.data)
                dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3, "offsets": [0, 4, 8], "itemsize": decoded.point_step})
                array = np.frombuffer(raw, dtype=dtype)
                xyz = np.stack([array[axis] for axis in ("x", "y", "z")], axis=1)
                clouds.append((stamp, xyz[np.isfinite(xyz).all(axis=1)]))

odom = np.asarray(odometry)
times = (odom[:, 0] - start_ns) / 1e9
interpolate_rotation = Slerp(times, Rotation.from_quat(odom[:, 4:8]))
pitch = Rotation.from_euler("x", args.pitch, degrees=True).as_matrix()
virtual_body_rotation = np.asarray(calibration["front_fisheye"]["T"]).reshape(4, 4)[:3, :3] @ pitch
W, H = args.width, args.height
focal = H / (2 * np.tan(np.deg2rad(args.fov) / 2))
K = np.asarray([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32)
records, front_poses, front_images = {}, [], []
for name in cameras:
    configuration = calibration[f"{name}_fisheye"]
    matrix = np.asarray(configuration["K"]).reshape(3, 3)
    distortion = np.asarray(configuration["D"])
    body_camera = np.asarray(configuration["T"]).reshape(4, 4)
    camera_virtual_rotation = body_camera[:3, :3].T @ virtual_body_rotation
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(matrix, distortion, camera_virtual_rotation.T, K, (W, H), cv2.CV_32FC1)
    validity = cv2.remap(np.ones((configuration["image_height"], configuration["image_width"]), np.uint8), map_x, map_y, cv2.INTER_NEAREST)
    folder = out / name
    folder.mkdir(exist_ok=True)
    cv2.imwrite(str(folder / "valid_mask.png"), validity * 255)
    bitstream = folder / "clip.h264"
    with bitstream.open("wb") as stream:
        for packet in packets[name]:
            stream.write(packet[2])
    capture = cv2.VideoCapture(str(bitstream))
    poses, camera_records = [], []
    with iio.get_writer(folder / "rectified.mp4", fps=10, codec="libx264", quality=8, macro_block_size=1) as movie:
        for index, packet in enumerate(packets[name]):
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode {name} frame {index}")
            image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
            cv2.imwrite(str(folder / f"{index:05d}.png"), image)
            movie.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            t = (packet[0] - start_ns) / 1e9
            body_pose = np.eye(4)
            body_pose[:3, :3] = interpolate_rotation(t).as_matrix()
            body_pose[:3, 3] = [np.interp(t, times, odom[:, axis]) for axis in (1, 2, 3)]
            virtual = np.eye(4)
            virtual[:3, :3] = virtual_body_rotation
            virtual[:3, 3] = body_camera[:3, 3]
            poses.append(body_pose @ virtual)
            camera_records.append({"frame": index, "capture_ns": packet[0], "log_ns": packet[1], "log_latency_ms": (packet[1] - packet[0]) / 1e6})
    capture.release()
    np.save(folder / "poses.npy", np.asarray(poses, dtype=np.float32))
    records[name] = camera_records

front_times = np.asarray([packet[0] for packet in packets["front"]])
front_cloud_times = np.asarray([cloud[0] for cloud in clouds])
camera_body = np.eye(4)
camera_body[:3, :3] = virtual_body_rotation
camera_body[:3, 3] = np.asarray(calibration["front_fisheye"]["T"]).reshape(4, 4)[:3, 3]
body_camera = np.linalg.inv(camera_body)
depth_dir = out / "lidar_depth"
depth_dir.mkdir(exist_ok=True)
offsets = []
for index, stamp in enumerate(front_times):
    closest = int(np.argmin(np.abs(front_cloud_times - stamp)))
    offsets.append(float((front_cloud_times[closest] - stamp) / 1e6))
    points = clouds[closest][1]
    np.savez_compressed(depth_dir / f"{index:05d}_points.npz", xyz_body_assumed=points)
    camera_xyz = points @ body_camera[:3, :3].T + body_camera[:3, 3]
    uv = camera_xyz @ K.T
    z = camera_xyz[:, 2]
    uv = np.rint(uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)).astype(int)
    mask = (z > .2) & (z < 20) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    depth = np.full((H, W), np.inf, np.float32)
    np.minimum.at(depth, (uv[mask, 1], uv[mask, 0]), z[mask])
    depth[np.isinf(depth)] = 0
    np.save(depth_dir / f"{index:05d}.npy", depth)
np.save(out / "odometry.npy", odom)
front_body_positions = np.asarray([np.interp((stamp - start_ns) / 1e9, times, odom[:, axis]) for stamp in front_times for axis in (1, 2, 3)]).reshape(-1, 3)
report = {"source_mcap": str(source), "start_capture_ns": int(start_ns), "start_offset_s": start_s, "duration_s": duration_s, "camera_records": records, "intrinsic": K.tolist(), "resolution": [W, H], "virtual_pitch_degrees": args.pitch, "vertical_fov_degrees": args.fov, "lidar_nearest_offsets_ms": offsets, "robot_translation_extent_m": np.ptp(front_body_positions, axis=0).tolist(), "calibration": calibration, "pose_assumption": "LIO map pose interpreted as body_FLU; camera T interpreted as camera-to-body. Front lidar is identity in embedded calibration; validate projection before using metric geometry."}
(out / "selection.json").write_text(json.dumps(report, indent=2))
print({"frames": {name: len(packets[name]) for name in cameras}, "robot_translation_extent_m": report["robot_translation_extent_m"], "lidar_offset_ms_range": [min(offsets), max(offsets)]}, flush=True)
