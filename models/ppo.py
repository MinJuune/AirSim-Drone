import torch
import os
from stable_baselines3 import PPO

import config.hyperparameter as hyperparameter
from utils.log import log
from models.multiModalEncoder import MultiModalEncoder


def train(env):
  log("DEBUG", f"Is GPU Usable: {torch.cuda.is_available()}")
  log("DEBUG", f"The Number Of GPUs: {torch.cuda.device_count()}")

  # 현재 시스템이 GPU(CUDA)를 사용할 수 있는지 확인하고, 학습에 사용할 장치 선택
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  log("INFO", f"Currently Using Device: {device}")
  if torch.cuda.is_available():
    log("INFO", f"Name Of The GPU: {torch.cuda.get_device_name(0)}")

  # PPO policy의 입력으로 사용할 상태 정보를 커스텀 방식으로 인코딩하는 클래스
  policy_kwargs = dict(
    features_extractor_class=MultiModalEncoder
  )

  # 기존 모델이 있으면 로드, 없으면 새 모델 생성
  if os.path.exists(hyperparameter.MODEL_PATH_PPO):
    log("INFO", "Loading The Existing Model...")
    model = PPO.load(hyperparameter.MODEL_PATH_PPO, env=env, device=device)
  else:
    log("INFO", "Generating A New Model...")
    model = PPO(hyperparameter.PPO_POLICY,
                env=env,
                verbose=1,
                n_steps=hyperparameter.PPO_N_STEPS,
                batch_size=hyperparameter.PPO_BATCH_SIZE,
                n_epochs=hyperparameter.PPO_N_EPOCHS,
                device=device,
                policy_kwargs=policy_kwargs)
  
  # 모델 학습 및 저장
  log("INFO", "Start Training The Model.")
  model.learn(total_timesteps=hyperparameter.TOTAL_TIMESTEPS)
  model.save(hyperparameter.MODEL_PATH_PPO)
  log("INFO", "Saved The Model.")