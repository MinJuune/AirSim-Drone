import keyboard
import airsim
import numpy as np

import config.config as config
import config.key as key
from utils.log import log


def manual_control(env):
    client = env.client
    _ = env.reset()
    log("INFO", f"This Is Manual Overriding Mode. Press {key.EXIT_KEY_MANUAL_OVERRIDING} To Exit.")

    while True:
        if keyboard.is_pressed(key.GO_FORWARD):
            client.moveByVelocityAsync(2, 0, 0, 1).join()
        elif keyboard.is_pressed(key.GO_BACK):
            client.moveByVelocityAsync(-2, 0, 0, 1).join()
        elif keyboard.is_pressed(key.GO_LEFT):
            client.moveByVelocityAsync(0, -2, 0, 1).join()
        elif keyboard.is_pressed(key.GO_RIGHT):
            client.moveByVelocityAsync(0, 2, 0, 1).join()
        elif keyboard.is_pressed(key.GO_UP):
            client.moveByVelocityAsync(0, 0, -2, 1).join()
        elif keyboard.is_pressed(key.GO_DOWN):
            client.moveByVelocityAsync(0, 0, 2, 1).join()
        elif keyboard.is_pressed(key.TURN_RIGHT):
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=20)
            client.moveByVelocityAsync(
            0, 0, 0,
            duration=1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode
            ).join()
        elif keyboard.is_pressed(key.TURN_LEFT):
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=-20)
            client.moveByVelocityAsync(
            0, 0, 0,
            duration=1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode
            ).join()
        elif keyboard.is_pressed(key.GET_STATE):
            _ = env._get_state()

            position = client.getMultirotorState().kinematics_estimated.position
            current_drone_position = np.array([position.x_val, position.y_val, position.z_val], dtype=np.float32)
            log("DEBUG", f"Current Drone Position: {current_drone_position}")

            #lidar_data = client.getLidarData(lidar_name=config.LIDAR_NAME)
            #visualize_lidar_feature(lidar_data)

        elif keyboard.is_pressed(key.EXIT_KEY_MANUAL_OVERRIDING):
            log("INFO", "Manual Overriding Mode Finished.")
            return