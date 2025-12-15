import config.config as config
from envs.environment import AirSimEnv
from models.ppo import train
from test import test
from utils.log import log
from utils.manualOverriding import manual_control


def main():
  # env 파일에서 정의한 환경의 객체 생성
  env = AirSimEnv()

  # test 모드일 경우
  if config.TEST_MODE:
    log("INFO", "This Is A Test Mode.")
    test(env)

  # manual overriding 모드일 경우
  elif config.MANUAL_OVERRIDING_MODE:
    log("INFO", "This Is A Manual Overriding Mode.")
    manual_control(env)

  # 기본 train 모드 
  else:
    log("INFO", "This Is A Train Mode.")
    train(env)
  
  # AirSim 환경 내부에서 사용한 리소스 정리 
  env.close()

# 이 파일이 직접 실행될때만 main()이 호출되도록
# 만약 그냥 main()인 경우 다른 파일에서 import 될때도 main() 실행됨
if __name__ == "__main__":
  main()