import numpy as np
import cv2
import airsim
import gym
from gym import spaces

import config.config as config
import config.reward as reward
from utils.log import log
from models.yolo import targetDetector
from utils.lidarPolarCoordinate import lidarPolarCoordinate
from utils.callback import visualize_lidar_points_unreal
from utils.image_save import ImageSaver
from utils.CalculatePCC import CalculatePCC

class AirSimEnv(gym.Env):
  def __init__(self):
    """
    AirSim 환경 초기화

    - AirSim 드론 클라이언트와 연결하고 제어 권한 설점
    - 관측공간(observation_space)과 행동공간(action_space)을 정의 
    """
    super().__init__()

    self.client = airsim.MultirotorClient()
    self.client.confirmConnection()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)

    self.yolo = targetDetector()
    self.image_saver = ImageSaver()
    self.calculate_pcc = CalculatePCC()

    self.prev_area_ratio = 0.0
    self.prev_center_offset_ratio = 0.0

    self.max_steps_in_episode_test=config.MAX_STEPS_IN_EPISODE_TEST

    self.observation_space = spaces.Dict({
      "lidar_points": spaces.Box(
        low=config.OBSERVATION_SPACE_LOW_LIDAR,
        high=config.OBSERVATION_SPACE_HIGH_LIDAR,
        shape=(config.NUM_OF_POINTS, config.DIM_OF_POINTS),
        dtype=np.float32
      ),
      "camera_image": spaces.Box(
        low=config.OBSERVATION_SPACE_LOW_CAMERA,
        high=config.OBSERVATION_SPACE_HIGH_CAMERA,
        shape=(config.CAMERA_HEIGHT, config.CAMERA_WIDTH, config.CAMERA_CHANNELS),
        dtype=np.uint8
      ),
      "yolo_info" : spaces.Box(
        low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        shape=(4,),
        dtype=np.float32
      ),
      "lidar_local_features": spaces.Box(
        low=0.0,
        high=1.0,
        shape=(1,36),
        dtype=np.float32
      )
    })

    self.action_space = spaces.Box(
        low=config.ACTION_SPACE_LOW,
        high=config.ACTION_SPACE_HIGH,
        shape=(4,),
        dtype=np.float32
    )

    log("INFO", "Initializing An Environment Completed.")
  
  def _get_state(self):
    """
    현재 상태(state)를 구성하여 반환

    Return:
      dict: 상태 정보를 담은 딕셔너리. 아래 키들을 포함:
        - "lidar_points"         : LiDAR 포인트 클라우드
        - "camera_image"         : 카메라 이미지
        - "yolo_info"            : YOLO 감지 정보 [탐지여부, 면적비율, 중심오차, 각도]
        - "lidar_local_features" : 극좌표 기반 라이다 특징 벡터 
    """
    # 1. "lidar_points" 정의 
    lidar_data = self.client.getLidarData(lidar_name=config.LIDAR_NAME)
    lidar_points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)

    num_points = lidar_points.shape[0]

    if num_points > config.NUM_OF_POINTS: # 2048개 보다 많으면 랜덤하게 제거
        indices = np.random.choice(num_points, config.NUM_OF_POINTS, replace=False)
        lidar_points = lidar_points[indices]
    elif num_points < config.NUM_OF_POINTS: # 2048개 보다 적으면 0으로 패딩 
        padding = np.zeros((config.NUM_OF_POINTS - num_points, 3), dtype=np.float32)
        lidar_points = np.vstack((lidar_points, padding))

    # 2. "camera_image" 정의
    image_response = self.client.simGetImages([
      airsim.ImageRequest(config.CAMERA_NAME, airsim.ImageType.Scene, pixels_as_float=False, compress=config.IMAGE_COMPRESSION)
    ])[0] # compress=True라서 JPEG 압축된 이미지가 반환

    if config.IMAGE_COMPRESSION:
      jpeg_bytes = image_response.image_data_uint8                  # JPEG 압축된 바이트 형식 이미지 
      jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)        # 바이트 데이터를 넘파이 배열로 변환
      camera_image = cv2.imdecode(jpeg_array, cv2.IMREAD_UNCHANGED) # opencv로 JPEG를 이미지로 복원
    else: # 압축안된 이미지는 그대로 이미지 픽셀 배열이라, 넘파이 배열로 바꾸고 shape만 지정
      camera_image = np.array(image_response.image_data_uint8, dtype=np.uint8).reshape(config.CAMERA_HEIGHT,
                                                                                       config.CAMERA_WIDTH,
                                                                                       config.CAMERA_CHANNELS)
    
    # 3. "yolo_info" 정의
    yolo_result = self.yolo.detect(camera_image[:, :, :3]) # (480,640,4)면 (480,640,3)으로 RGB만. 원래는 RGBA(A는 투명도)
    yolo_info_array = np.array([
      float(yolo_result["detected"]), 
            yolo_result["area_ratio"],
            yolo_result["center_offset_ratio"],
            yolo_result["theta_norm"]
    ], dtype=np.float32)

    # 4. "lidar_local_features" 정의
    lidar_local_features = lidarPolarCoordinate(lidar_points)
    print(f"***********+++ {lidar_points}")

    state = {
      "lidar_points": lidar_points,
      "camera_image": camera_image,
      "yolo_info": yolo_info_array, 
      "lidar_local_features": lidar_local_features
    }

    log("DEBUG", f"LiDAR shape: {lidar_points.shape}, Image shape: {camera_image.shape}, YOLO: {yolo_result}")
    return state
  
  def reset(self):
    """
    환경을 초기화하고 드론을 이륙시킨 뒤 시작 위치로 이동

    Return: 
      dict: 초기 상태 정보
    """
    self.client.reset()
    self.client.enableApiControl(True)
    self.client.armDisarm(True)

    self.client.takeoffAsync().join()
    self.client.moveToPositionAsync(*config.START_POS).join()

    self.steps_in_episode = 0

    obs = self._get_state()

    log("DEBUG", "Initializing The Environment Completed.")
    return obs
  
  def close(self):
    """
    드론의 제어를 해제하고 환경을 종료
    """
    self.client.armDisarm(False)
    self.client.enableApiControl(False)
    self.client.reset()
    log("INFO", "The Environment Closed.")
  
  def calculate_reward(self, collision_info, prev_distance, current_distance, current_drone_position, yolo_info, vy, lidar_points):
    """
    현재 상태를 기반으로 보상을 계산

    Arg:
      collision_info : 충돌 여부 및 충돌 정보
      prev_distance : 이전 위치에서 목표까지의 거리 
      current_distance : 현재 위치에서 목표까지의 거리
      current_drone_position : 현재 드론의 좌표
      yolo_info : YOLO 탐지 정보 [탐지여부, 면적비율, 중심오차, 각도]

    Return:
      tuple : 계산된 보상값과 에피소드 종료 여부  

    보상 로직:
      충돌했을 때                -> 패널티 부여 후 종료
      최대 스템 수를 초과했을 때 -> 패널티 부여 후 종료
      목표 지점에 도달했을 때    -> 보상 부여 후 종료 
      안전 구역을 벗어났을 때    -> 패널티 부여 후 종료
      
      목표에 가까워졌을 때         -> 보상 
      목표에서 먹어졌을 때         -> 패널티 
      타겟을 탐지했을 때           -> 보상 
      타겟을 탐지하지 못했을 때    -> 패널티
      바운딩 박스 크기 커졌을 때   -> 보상
      바운딩 박스 크기 작아졌을 때 -> 패널티
      중심에 가까워졌을 때         -> 보상
      중심에서 멀어졌을 때         -> 패널티
    """
    if collision_info.has_collided:
      log("INFO", "A Collision Occured.")
      self.steps_in_episode = 0
      return reward.REWARD_COLLISION, True
    
    if self.steps_in_episode >= config.MAX_STEPS_IN_EPISODE_TRAIN:
      log("INFO", "Max Steps Exceed.")
      self.steps_in_episode = 0
      return reward.REWARD_MAX_STEPS_EXCEED, True
    
    if current_distance < config.GOAL_TOLERANCE:
      log("INFO", "Reached The Goal.")
      self.steps_in_episode = 0
      return reward.REWARD_GOAL, True
    
    center = config.SAFE_BOUND_CENTER
    distance_from_center = np.linalg.norm(current_drone_position - center)
    if distance_from_center > config.SAFE_BOUND:
      log("INFO", "Leaved The Safe Bound.")
      self.steps_in_episode = 0
      return reward.REWARD_OUT_OF_BOUNDS, True
    
    calculated_reward = reward.REWARD_DEFAULT

    delta_distance = prev_distance - current_distance
    if delta_distance > 0:
      calculated_reward += reward.REWARD_DISTANCE_GAIN * (delta_distance ** 1.5)
      log("DEBUG", "Get Closer To The Goal.")
    else:
      calculated_reward += reward.REWARD_DISTANCE_LOSS * (abs(delta_distance) ** 1.5)
      log("DEBUG", "Get Away From The Goal.")

    detected, area_ratio, center_offset_ratio, _ = yolo_info

    if detected == 1.0:
      calculated_reward += reward.YOLO_DETECTED
    else:
      calculated_reward += reward.YOLO_NOT_DETECTED
      calculated_reward += abs(vy) * reward.VY_COEFFICIENT

    delta_area = area_ratio - self.prev_area_ratio
    calculated_reward += reward.YOLO_AREA_COEFFICIENT * delta_area

    delta_center_offset = center_offset_ratio - self.prev_center_offset_ratio
   
    calculated_reward += reward.YOLO_CENTER_OFFSET_COEFFICIENT * delta_center_offset

    self.steps_in_episode += 1

    # LiDAR 거리 기반 패널티
    if lidar_points.shape[0] > 0:
      lidar_xy = lidar_points[:, :2]  # (x,y)
      distances = np.linalg.norm(lidar_xy, axis=1)
      min_dist = np.min(distances)

      if min_dist < config.LIDAR_OBSTACLE_THRESHOLD:
        penalty = reward.LIDAR_OBSTACLE_PENALTY * (config.LIDAR_OBSTACLE_THRESHOLD - min_dist)
        calculated_reward += penalty
        log("DEBUG", f"Obstacle too close! LiDAR min_dist = {min_dist:.2f}, penalty = {penalty:.2f}")


    print(f"calculated_reward: {calculated_reward}")
    return calculated_reward, False
      
  def step(self, action):
    """
    주어진 action을 환경에 적용하고 다음 상태,보상,종료 여부를 반환

    Args:
      action : 드론 제어 명령(vx,vy,vz,yaw)

    Return:
      tuple : 다음 상태, 보상, 종료여부
    """

    # 보상 계산을 위한 이전 상태 기록
    prev_pos = self.client.getMultirotorState().kinematics_estimated.position
    prev_drone_position = np.array([prev_pos.x_val, prev_pos.y_val, prev_pos.z_val], dtype=np.float32)
    # calculate_reward 2번째 인자
    prev_distance = np.linalg.norm(prev_drone_position - config.TARGET_POS)

    # calculate_reward 1번째 인자
    collision_info = self.client.simGetCollisionInfo()

    # action을 환경에 적용
    vx, vy, vz, yaw = map(float, action)
    if (config.Z_FREEZE == True):
      vz = 0
      
    log("DEBUG", "== Action ==")
    log("DEBUG", f"vx = {vx}, xy = {vy}, vz = {vz}, yaw = {10 * yaw}")

    # 1. obs 정의 (여기서 또 쓰는 이유는 action 이후의 상태를 얻기 위함)
    obs = self._get_state()

    yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=config.YAW_COEFFICIENT*yaw)
    self.client.moveByVelocityAsync(
      vx, vy, vz,
      duration=config.ACTION_DURATION,
      drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
      yaw_mode=yaw_mode
    ).join()

    # 시각화 및 YOLO 정보 추출
    lidar_points = obs["lidar_points"]
    visualize_lidar_points_unreal(lidar_points, self.client)
    # calculate_reward 5번째 인자
    yolo_info = obs["yolo_info"]

    # 보상 계산을 위한 현재 위치 정보 수집
    position = self.client.getMultirotorState().kinematics_estimated.position
    # calculate_reward 4번째 인자
    current_drone_position = np.array([position.x_val, position.y_val, position.z_val], dtype=np.float32)
    # calculate_reward 3번째 인자
    current_distance = np.linalg.norm(current_drone_position - config.TARGET_POS)
    
    # 2,3 calculated_reward, done 정의 
    log("INFO", f"Current Distance: {current_distance}")
    calculated_reward, done = self.calculate_reward(collision_info, prev_distance, current_distance, current_drone_position, yolo_info, vy, lidar_points)

    # YOLO 상태 저장(다음 스텝 대비)
    self.prev_area_ratio = yolo_info[1]
    self.prev_center_offset = yolo_info[2]

    # 이미지 저장 및 상관계수 기록
    camera_image = obs["camera_image"]
    self.image_saver.save(camera_image)
    self.calculate_pcc.log(center_offset=yolo_info[2], theta_norm=yolo_info[3])

    log("INFO", f"reward: {calculated_reward:.2f}")
    return obs, calculated_reward, done, {}