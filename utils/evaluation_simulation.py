import numpy as np

# 1. 성공률
def compute_success_rate(successes, total):
    """
    성공률 계산 함수
    
    Args:
        successes (int): 성공한 에피소드 수
        total (int): 전체 에피소드 수

    Returns:
        float: 성공률 (%)
    """
    if total == 0:
        return 0.0
    return (successes / total) * 100


# 2. 평균 에피소드 보상
def compute_average_reward(reward_list):
    """
    평균 에피소드 보상 계산 함수

    Args:
        reward_list (List[float]): 각 에피소드마다의 총 보상 리스트

    Returns:
        float: 평균 보상
    """
    if len(reward_list) == 0:
        return 0.0
    return sum(reward_list) / len(reward_list)

# 3. 학습 수렴도 
def compute_convergence_rate(avg_rewards, r_target):
    """
    학습 수렴도 계산 함수

    Args:
        avg_rewards (List[float]): 학습 반복별 평균 보상 리스트
        r_target (float): 수렴 기준 임계값

    Returns:
        int or None: 수렴까지 걸린 반복 수 (수렴하지 않으면 None)
    """
    for k, r_k in enumerate(avg_rewards):
        if r_k >= r_target:
            return k
    return None

# 4. 학습 안정성 (Training Stability)
def compute_training_stability(average_rewards: list[float]) -> float:
    """
    학습 안정성: 보상 분산
    """
    if len(average_rewards) == 0:
        return 0.0
    return float(np.var(average_rewards))

# 5. 표본 효율 (Sample Efficiency)
def compute_sample_efficiency(average_rewards: list[float], target_reward: float) -> int:
    """
    목표 보상에 도달하는 데 걸린 최소 학습 반복 수 반환 (0-based index + 1)
    """
    for i, r in enumerate(average_rewards):
        if r >= target_reward:
            return i + 1
    return len(average_rewards)

"""
테스트
"""

# 1. 성공률
def compute_test_success_rate(success_count: int, total_test_episodes: int) -> float:
    """
    테스트 환경에서 목표에 도달한 비율
    """
    if total_test_episodes == 0:
        return 0.0
    return (success_count / total_test_episodes) * 100

# 2. 평균 목표 오차 (평균 유클리드 거리)
def compute_average_goal_error(failed_final_positions: list[np.ndarray], goal_position: np.ndarray) -> float:
    """
    실패한 에피소드들의 최종 위치와 목표 위치 간 평균 거리
    """
    if len(failed_final_positions) == 0:
        return 0.0
    distances = [np.linalg.norm(p - goal_position) for p in failed_final_positions]
    return float(np.mean(distances))

# 3. 충돌률 & 안전성
def compute_collision_and_safety_rate(collision_count: int, total_test_episodes: int) -> tuple[float, float]:
    """
    충돌률과 안전성 반환 (% 단위)
    """
    if total_test_episodes == 0:
        return 0.0, 100.0
    collision_rate = (collision_count / total_test_episodes) * 100
    safety_rate = 100.0 - collision_rate
    return collision_rate, safety_rate

# 4. 경로 효율성
def compute_path_efficiency(actual_distance: float, shortest_distance: float) -> float:
    """
    최단 거리 대비 실제 이동 거리의 비율 (1.0에 가까울수록 좋음)
    """
    if actual_distance == 0:
        return 0.0
    return shortest_distance / actual_distance

# 5. 평균 속도 및 도달 시간
def compute_average_velocity_and_duration(distances: list[float], durations: list[float]) -> tuple[float, float]:
    """
    평균 속도와 평균 소요 시간 반환
    """
    if len(distances) == 0 or len(durations) == 0:
        return 0.0, 0.0
    avg_velocity = np.mean([d / t if t > 0 else 0.0 for d, t in zip(distances, durations)])
    avg_duration = np.mean(durations)
    return float(avg_velocity), float(avg_duration)
