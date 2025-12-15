import numpy as np
import os
import keyboard
from stable_baselines3 import PPO

from envs.environment import AirSimEnv
from utils.log import log
from utils.manualOverriding import manual_control
from utils.callback import testResult
import config.hyperparameter as hyperparameter
import config.config as config
import config.key as key


def test(env):
  # 저장된 PPO 모델 없으면 에러 
  if not os.path.exists(hyperparameter.MODEL_PATH_PPO):
    log("ERROR", "There Is No Model.")
    return
  
  # PPO 모델 불러오기 
  log("INFO", "Loading Saved Model.")
  model = PPO.load(hyperparameter.MODEL_PATH_PPO, env=env)

  log("INFO", "Start Testing Mode")
  log("INFO", f"Press {key.EXIT_KEY_MANUAL_OVERRIDING} To Manual Overriding Mode.")
  log("INFO", f"Press {key.EXIT_KEY_TEST} TO Quit.")

  # 결과 저장용 리스트트 초기화
  all_trajectory = []         # 드론의 경로를 담는 좌표 리스트
  episode_steps = []          # 각 에피소드에서 실행된 스텝 수 
  episode_total_rewards = []  # 에피소드의 누적된 보상 총합

  # 최대 테스트 횟수만큼 에피소드 반복 
  for episode in range(config.TEST_MAX_EPISODES):
    # 매 에피소드마다 환경 초기화, 누적 보상 초기화
    obs = env.reset()
    total_reward = 0

    # 위치 기록용 리스트 
    positions_x = []
    positions_y = []
    positions_z = []

    # 각 에피소드 내에서 정해진 스텝 수만큼 행동 반복
    for step in range(config.TEST_EPISODE_STEPS):
      # 수동 모드로 전환
      if keyboard.is_pressed(key.EXIT_KEY_MANUAL_OVERRIDING):
        manual_control(env)
        return
      
      # 학습된 PPO 모델이 현재 상태 obs를 보고 action을 선택
      action, _ = model.predict(obs)          

      # 해당 action을 환경에 적용하고 다음 상태, 보상, 종료 여부 받음
      obs, reward, done, _ = env.step(action) 

      # 위치 기록 및 보상 계산
      pos = env.client.getMultirotorState().kinematics_estimated.position
      drone_position = np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

      x, y, z = drone_position
      positions_x.append(x)
      positions_y.append(y)
      positions_z.append(z)

      # 누적 보상 계산
      total_reward += reward

      # 에피소드, 스텝, 보상, 위치 출력
      log("DEBUG", f"Episode: {episode+1} | Step: {step} | Reward: {reward:.2f} | Position: ({x:.2f}, {y:.2f}, {z:.2f})")

      if done:
        break
    
    # 에피소드 기록 저장
    all_trajectory.append((positions_x, positions_y, positions_z))
    episode_total_rewards.append(total_reward)
    episode_steps.append(step + 1)

    # utils > callback.py의 함수
    testResult(env, episode_total_rewards, episode_steps, all_trajectory)