import matplotlib.pyplot as plt
import numpy as np
import airsim

import config.config as config
from utils.log import log


def testResult(env, episode_total_rewards, episode_steps, all_trajectory):
    log("INFO", "\n=== Result Of The Test ===")
    log("INFO", f"Rewards: {episode_total_rewards}")
    log("INFO", f"Total Rewards: {np.sum(episode_total_rewards):.2f}")
    log("INFO", f"Mean Of Total Rewards: {np.mean(episode_total_rewards):.2f}")
    log("INFO", f"Mean Of Steps Per Episode: {np.mean(episode_steps):.2f}")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, (x_vals, y_vals, z_vals) in enumerate(all_trajectory):
        ax.plot(x_vals, y_vals, z_vals, marker='o', linestyle='-', label=f'Episode {i+1}')

    # 이거 config.TARGET_POS임?
    target_x, target_y, target_z = config.TARGET_POS
    ax.scatter(target_x, target_y, target_z, color='r', marker='X', s=100, label='Target')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Z Position")
    ax.set_title("Flight Path of Drone")
    ax.legend()
    plt.show()



def visualize_lidar_points_unreal(lidar_points, client):
    if lidar_points is None or len(lidar_points) == 0:
        print("LiDAR 데이터가 없습니다.")
        return

    airsim_points = [airsim.Vector3r(float(p[0]), float(p[1]), float(p[2])) for p in lidar_points]
    client.simPlotPoints(
        points=airsim_points,
        color_rgba=[0, 255, 0, 255],
        size=5.0,
        duration=0.5
    )
