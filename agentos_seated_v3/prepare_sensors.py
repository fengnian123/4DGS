import json
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'agentos_seated'
V2 = ROOT.parent / 'agentos_seated_v2'


class Sensors:
    def __init__(self):
        self.selection = json.loads((BASE / 'clip/selection.json').read_text())
        self.origin = self.selection['start_capture_ns'] / 1e9
        self.cameras = {name: np.array([r['capture_ns'] for r in rows], np.int64) / 1e9 - self.origin
                        for name, rows in self.selection['camera_records'].items()}
        self.K = np.array(self.selection['intrinsic'])
        self.calibration = self.selection['calibration']
        self.clouds = {'front': [], 'rear': []}
        odom, imu = [], []
        factory = DecoderFactory()
        with (BASE / 'clip/selected.mcap').open('rb') as stream:
            topics = ['/sensor/lidar/front_lidar_pcd', '/sensor/lidar/rear_lidar_pcd', '/LIO_ODOM_HIGH_FREQUENCY', '/IMU']
            for schema, channel, message in make_reader(stream).iter_messages(topics=topics):
                obj = factory.decoder_for(channel.message_encoding, schema)(message.data)
                stamp = obj.header.stamp.sec + obj.header.stamp.nanosec / 1e9 - self.origin
                if 'lidar' in channel.topic:
                    name = 'front' if 'front' in channel.topic else 'rear'
                    fields = {f.name: f for f in obj.fields}
                    dtype = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring', 'timestamp'],
                                      'formats': ['<f4'] * 4 + ['<u2', '<f8'],
                                      'offsets': [fields[k].offset for k in ('x', 'y', 'z', 'intensity', 'ring', 'timestamp')],
                                      'itemsize': obj.point_step})
                    assert not obj.is_bigendian and obj.height == 1
                    array = np.frombuffer(bytes(obj.data), dtype=dtype)
                    xyz = np.stack([array[k] for k in ('x', 'y', 'z')], axis=1)
                    times = array['timestamp'] - self.origin
                    valid = np.isfinite(xyz).all(1) & np.isfinite(times) & (np.linalg.norm(xyz, axis=1) > .25)
                    self.clouds[name].append({'header': stamp, 'xyz': xyz[valid], 'times': times[valid],
                                              'intensity': array['intensity'][valid], 'ring': array['ring'][valid]})
                elif 'ODOM' in channel.topic:
                    p, q = obj.pose.pose.position, obj.pose.pose.orientation
                    odom.append([stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w])
                else:
                    w = obj.angular_velocity
                    imu.append([stamp, w.x, w.y, w.z])
        self.odom, self.imu = np.array(odom), np.array(imu)
        self.slerp = Slerp(self.odom[:, 0], Rotation.from_quat(self.odom[:, 4:8]))
        self.common_rotation = np.array(self.calibration['front_fisheye']['T']).reshape(4, 4)[:3, :3]
        self.body_cameras = {}
        for name in self.cameras:
            pose = np.eye(4)
            pose[:3, :3] = self.common_rotation
            pose[:3, 3] = np.array(self.calibration[name + '_fisheye']['T']).reshape(4, 4)[:3, 3]
            self.body_cameras[name] = pose

    def body_pose(self, times):
        times = np.asarray(times).reshape(-1).clip(self.odom[0, 0], self.odom[-1, 0])
        rotations = self.slerp(times).as_matrix()
        positions = np.stack([np.interp(times, self.odom[:, 0], self.odom[:, i]) for i in (1, 2, 3)], axis=1)
        return rotations, positions

    def points(self, time, radius=.025, names=('front', 'rear'), deskew=True):
        xyz, ts, headers, ids = [], [], [], []
        for sensor, name in enumerate(names):
            for packet_index, packet in enumerate(self.clouds[name]):
                keep = np.abs(packet['times'] - time) <= radius
                if not keep.any():
                    continue
                xyz.append(packet['xyz'][keep]); ts.append(packet['times'][keep])
                headers.append(np.full(keep.sum(), packet['header']))
                ids.append(sensor * 100000000 + packet_index * 100000 + np.flatnonzero(keep))
        xyz, ts, headers, ids = map(np.concatenate, (xyz, ts, headers, ids))
        if deskew:
            r, p = self.body_pose(ts)
            world = np.einsum('nij,nj->ni', r, xyz) + p
            r0, p0 = self.body_pose([time])
            xyz = (world - p0[0]) @ r0[0]
        return xyz.astype(np.float32), ts, ids, headers

    def project(self, xyz, camera='front'):
        transform = np.linalg.inv(self.body_cameras[camera])
        camera_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
        projection = camera_xyz @ self.K.T
        uv = np.rint(projection[:, :2] / projection[:, 2:].clip(1e-5)).astype(int)
        z = camera_xyz[:, 2]
        keep = (z > .12) & (z < 12) & (uv[:, 0] >= 0) & (uv[:, 0] < 768) & (uv[:, 1] >= 0) & (uv[:, 1] < 512)
        return uv, z, keep


def raster(uv, z, valid):
    depth = np.full((512, 768), np.inf, np.float32)
    np.minimum.at(depth, (uv[valid, 1], uv[valid, 0]), z[valid])
    depth[~np.isfinite(depth)] = 0
    return depth


def region_stats(depth, region):
    measured = (depth > 0) & region
    expanded = cv2.dilate((depth > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return {'pixels': int(region.sum()), 'measured_pixels': int(measured.sum()),
            'pixel_coverage': float(measured.sum() / max(1, region.sum())),
            'coverage_within_2px': float((expanded & region).sum() / max(1, region.sum())),
            'depth_median_m': float(np.median(depth[measured])) if measured.any() else None}


def audit_coordinates(sensors):
    import open3d as o3d
    def static_points(frame):
        time = sensors.cameras['front'][frame]
        xyz, ts, ids, headers = sensors.points(time, radius=.025)
        uv, z, keep = sensors.project(xyz)
        mask = cv2.dilate(cv2.imread(str(V2 / f'agentos_office/masks/{frame:05d}.png'), 0), np.ones((15, 15), np.uint8)) > 0
        keep &= ~mask[uv[:, 1].clip(0, 511), uv[:, 0].clip(0, 767)]
        return xyz[keep], ts[keep], headers[keep]
    xyz, times, headers = static_points(40)
    r, p = sensors.body_pose([sensors.cameras['front'][40]])
    target_world = xyz @ r[0].T + p[0]
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_world)).voxel_down_sample(.025)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.12, max_nn=30))
    target_xyz, normals = np.asarray(target.points), np.asarray(target.normals)
    tree = cKDTree(target_xyz)
    records = []
    for frame in (0, 10, 20, 60, 79):
        # Deskewed points are in the camera-time body frame, hence one body pose here.
        xyz, _, _ = static_points(frame)
        time = sensors.cameras['front'][frame]
        r, p = sensors.body_pose([time])
        world = xyz @ r[0].T + p[0]
        inverse = (xyz - p[0]) @ r[0]
        r40, p40 = sensors.body_pose([sensors.cameras['front'][40]])
        frozen = xyz @ r40[0].T + p40[0]
        row = {'frame': frame}
        for name, values in (('lio_map_from_body', world), ('wrong_inverse_pose', inverse), ('ignore_robot_motion', frozen)):
            distance, idx = tree.query(values, workers=4)
            valid = distance < .15
            error = np.abs(((values - target_xyz[idx]) * normals[idx]).sum(1))
            row[name] = {'overlap_fraction': float(valid.mean()), 'median_plane_residual_m': float(np.median(error[valid])) if valid.any() else None}
        records.append(row)
    return records


def main():
    cv2.setNumThreads(4)
    destination = ROOT / 'sensors'
    destination.mkdir(exist_ok=True)
    sensors = Sensors()
    timing = {}
    for name, packets in sensors.clouds.items():
        offsets = np.concatenate([packet['times'] - packet['header'] for packet in packets])
        points = np.concatenate([packet['xyz'][::20] for packet in packets])
        timing[name] = {'packets': len(packets), 'point_minus_header_ms_quantiles': (np.quantile(offsets, [0, .1, .5, .9, 1]) * 1000).tolist(),
                        'xyz_1_50_99_percentiles_m': np.quantile(points, [.01, .5, .99], axis=0).tolist()}
    lio = Rotation.from_quat(sensors.odom[:, 4:8])
    dt = np.diff(sensors.odom[:, 0])
    omega = (lio[:-1].inv() * lio[1:]).as_rotvec() / dt[:, None]
    midpoint = (sensors.odom[:-1, 0] + sensors.odom[1:, 0]) / 2
    imu = np.stack([np.interp(midpoint, sensors.imu[:, 0], sensors.imu[:, i]) for i in (1, 2, 3)], axis=1)
    timing['lio_imu_angular_rate_median_difference_rad_s'] = float(np.median(np.linalg.norm(omega - imu, axis=1)))
    coverage = []
    for frame, time in enumerate(sensors.cameras['front']):
        xyz, ts, ids, headers = sensors.points(time)
        uv, z, keep = sensors.project(xyz)
        # Split by original point identity, before projection, so a held-out return never enters training.
        heldout = ((ids * 2654435761) % 100) < 20
        # Every camera and frame uses the same identity split.
        for camera in ('front', 'right', 'left'):
            uv, z, keep = sensors.project(xyz, camera)
            valid_fov = cv2.imread(str(BASE / f'clip/{camera}/valid_mask.png'), 0) > 0
            keep &= valid_fov[uv[:, 1].clip(0, 511), uv[:, 0].clip(0, 767)]
            train = raster(uv, z, keep & ~heldout)
            test = raster(uv, z, keep & heldout)
            overlap = (train > 0) & (test > 0)
            test[overlap] = 0
            folder = destination / camera
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f'{frame:05d}.npz', train_depth=train, heldout_depth=test)
            if camera == 'front':
                mask = cv2.imread(str(V2 / f'agentos_office/masks/{frame:05d}.png'), 0) > 0
                prop = cv2.imread(str(V2 / f'prop_masks/{frame:05d}.png'), 0) > 0
                full = raster(uv, z, keep)
                old = np.load(BASE / f'clip/lidar_depth/{frame:05d}.npy')
                row = {'frame': frame, 'selected_point_time_radius_ms': 25,
                       'actual_point_minus_camera_ms_quantiles': ((np.quantile(ts, [0, .5, 1]) - time) * 1000).tolist(),
                       'foreground': region_stats(full, mask), 'prop': region_stats(full, prop),
                       'old_foreground': region_stats(old, mask), 'old_prop': region_stats(old, prop),
                       'train_pixels': int((train > 0).sum()), 'heldout_pixels': int((test > 0).sum())}
                coverage.append(row)
                if frame in (0, 20, 30, 40, 48, 60, 79):
                    rgb = cv2.imread(str(BASE / f'clip/front/{frame:05d}.png'))
                    colors = cv2.applyColorMap((np.clip(full / 4, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    overlay = rgb.copy()
                    for y, x in zip(*np.where(full > 0)):
                        cv2.circle(overlay, (int(x), int(y)), 1, colors[y, x].tolist(), -1)
                    cv2.imwrite(str(destination / f'overlay_{frame:05d}.jpg'), overlay)
        if frame % 10 == 0:
            print('Sensor projection', frame, coverage[-1]['foreground'], coverage[-1]['prop'], flush=True)
    report = {'timing': timing, 'coverage': coverage, 'coordinate_diagnostics': audit_coordinates(sensors),
              'conventions': {'calibration_reference': sensors.calibration['reference_frame'],
                              'camera_T': 'camera optical to body FLU; axes and projected overlays checked',
                              'lidar_T': 'embedded identities; front positive x and rear negative x support body-aligned outputs, not an independent extrinsic calibration',
                              'lio': 'map-from-body convention checked against static point-plane consistency and IMU angular rate',
                              'time': 'camera capture stamp and absolute per-return LiDAR stamp, not message log/header as exposure time',
                              'deskew': 'rigid robot motion compensation from per-return time to camera time; no compensation for human motion',
                              'remaining_uncertainty': 'physical camera exposure delay and undocumented upstream pointcloud deskew are not independently identifiable from this short clip'}}
    (ROOT / 'sensor_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'timing': timing, 'coordinate_diagnostics': report['coordinate_diagnostics']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
