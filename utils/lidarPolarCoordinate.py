import torch as th
import numpy as np
import config.config as config

def angle_diff(a, b):
    diff = np.abs(a - b)
    return np.minimum(diff, 360 - diff)

def lidarPolarCoordinate(lidar_data):
    # point_cloud = np.array(lidar_data.point_cloud, dtype=np.float32)
    if lidar_data.size == 0:
        return np.ones(int(360 / config.LIDAR_ANGLE), dtype=np.float32)

    print(f"0. ++{lidar_data}")
    print(f"1. ++{lidar_data[:, 0]}")
    print(f"1. ++{lidar_data[:, 1]}")
    # points = point_cloud.reshape(-1, 3)
    x, y = lidar_data[:, 0], lidar_data[:, 1]
    theta_deg = np.degrees(np.arctan2(y, x)) % 360

    print(f"2. ++{theta_deg}")
    angle_bins = np.arange(0, 360, config.LIDAR_ANGLE)

    print(f"3. ++{angle_bins}")
    lidar_feature = np.ones(len(angle_bins), dtype=np.float32)
    print(f"4. ++{lidar_feature}")

    for i, angle in enumerate(angle_bins):
        mask = angle_diff(theta_deg, angle) < config.LIDAR_ANGLE_TOLERANCE
        bin_points = lidar_data[mask]

        if bin_points.shape[0] != 0:
            dists = np.linalg.norm(bin_points[:, :2], axis=1)
            min_dist = np.min(dists)
            norm_dist = np.clip(min_dist / config.LIDAR_MAX_RANGE, 0.0, 1.0)
            lidar_feature[i] = norm_dist

    lidar_feature = np.expand_dims(lidar_feature, axis=0)
    lidar_feature = th.from_numpy(lidar_feature).float()

    return lidar_feature
