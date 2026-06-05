# G1 + RUKA 다단계 강화학습 보상함수 설계 및 적용 계획

사용자의 요청에 따라 단순 도달(Reach)에 국한되어 있던 기존 보상 설계를 확장하여, **오브젝트 도달 -> 파지 및 들어 올리기(Grasp & Lift) -> 허리 앞 5cm 지점에 놓기(Place)**까지 순차적으로 수행할 수 있는 다단계(Stage-based) 보상 시스템을 구현합니다.

## User Review Required

> [!IMPORTANT]
> - **단계적 보상 활성화 기법(Stage-based Masking)**을 도입합니다. 도달이 먼저 성공해야 들어 올리기 보상이 제공되고, 들어 올리기가 성공해야 최종 목표 지점(허리 앞 5cm) 도달 보상이 제공됩니다. 이렇게 함으로써 강화학습 에이전트가 단계를 건너뛰어 학습이 무너지는 현상(예: 우연히 물체가 튕겨 나가서 목표점에 도달하는 등)을 원천 차단합니다.
> - **허리(waist/pelvis) 기준 좌표 설정**: 로봇 골반(`pelvis`)의 현재 위치를 기준으로 앞쪽(+x축 방향) 5cm, 높이 +15cm 지점을 최종 플레이스 홀더 목표 위치로 자동 계산하도록 구성합니다.

## Proposed Changes

### [RL Environment]

#### [MODIFY] [g1_ruka_env_cfg.py](file:///c:/isaac/isaac-sim-project-with-ruka/scripts/g1_ruka_env_cfg.py)

- `object_hand_distance` 대신에 아래와 같이 구성된 다단계 보상 계산 함수들을 추가/수정합니다.
  1. `ruka_Palm_Link`(손바닥)의 body index를 동적으로 탐색하여 손-큐브 거리를 구합니다.
  2. `pelvis`(허리)의 body index를 동적으로 탐색하여 **허리 앞 5cm 지점**(`[0.05, 0.0, 0.15]` offset)을 목표점으로 계산합니다.
  3. **Reach Reward**: 손바닥과 큐브 간의 거리가 감소하면 증가하는 $\tanh$ 보상.
  4. **Lift Reward**: 손이 큐브에 가까운 조건(예: 거리 < 10cm)에서 큐브의 높이가 바닥(초기 z=0.0375)보다 들려 올라갈수록 보상 부여.
  5. **Place Reward**: 큐브가 충분히 들린 조건(예: 들린 높이 > 5cm)에서 큐브의 위치가 목표 지점(허리 앞 5cm)과 가까워질수록 보상 부여.

```python
# 수정 및 적용할 보상 계산 함수 예시
def multi_stage_manipulation_rewards(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> dict:
    # 1. 에셋 데이터 참조
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    
    # 2. 바디 인덱스 탐색
    palm_id, _ = robot.find_bodies("ruka_Palm_Link")
    pelvis_id, _ = robot.find_bodies("pelvis")
    
    # 3. 위치 정보 획득
    palm_pos = robot.data.body_pos_w[:, palm_id[0]]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_id[0]]
    obj_pos = obj.data.root_pos_w
    
    # 4. 목표 배치 위치 계산 (허리 앞 5cm, 높이 +15cm)
    # 로봇의 앞쪽이 +x축이므로, pelvis 위치에서 x축 방향으로 +0.05m 오프셋 적용
    target_place_pos = pelvis_pos + torch.tensor([0.05, 0.0, 0.15], device=env.device)
    
    # 5. 각 거리 측정
    dist_hand_obj = torch.norm(obj_pos - palm_pos, dim=1)
    dist_obj_target = torch.norm(obj_pos - target_place_pos, dim=1)
    
    # 6. 단계별 상태 조건 정의
    # 손바닥이 오브젝트 10cm 이내로 접근했는가
    is_reached = (dist_hand_obj < 0.10).float()
    
    # 오브젝트가 초기 높이(0.0375m) 대비 5cm 이상 들렸는가
    lift_height = obj_pos[:, 2] - 0.0375
    is_lifted = (lift_height > 0.05).float()
    
    # 7. 단계별 보상 수식
    # (1) 도달 보상
    reward_reach = 1.0 - torch.tanh(dist_hand_obj / 0.15)
    
    # (2) 파지 및 들어올리기 보상 (손이 도달했을 때만 제공)
    reward_lift = is_reached * torch.clamp(lift_height * 10.0, max=1.0)
    
    # (3) 허리 앞 5cm에 두기 보상 (오브젝트가 충분히 들렸을 때만 제공)
    reward_place = is_lifted * (1.0 - torch.tanh(dist_obj_target / 0.15))
    
    # 보상들의 조합 반환
    return {
        "reach": reward_reach,
        "lift": reward_lift,
        "place": reward_place,
    }
```

- `RewardsCfg`에서 기존 `object_hand_distance`를 주석 처리하고, 다음과 같이 3개의 신규 보상 항목을 등록합니다:
  ```python
  # RewardsCfg 내 추가
  reward_reach = RewTerm(
      func=lambda env: multi_stage_manipulation_rewards(env)["reach"],
      weight=5.0,
  )
  reward_lift = RewTerm(
      func=lambda env: multi_stage_manipulation_rewards(env)["lift"],
      weight=15.0,
  )
  reward_place = RewTerm(
      func=lambda env: multi_stage_manipulation_rewards(env)["place"],
      weight=20.0,
  )
  ```

- 학습의 편의를 위해 `G1RukaEnvCfg`의 에피소드 길이(`episode_length_s`)를 `5.0초`에서 `8.0초`로 소폭 상향 조정을 제안합니다 (도달 후 파지 및 배치까지 긴 스텝이 필요할 수 있기 때문).

## Verification Plan

### Automated Tests
- 수정된 설정 파일이 정상적으로 임포트되고 에러가 없는지 검증하기 위해 play_ppo 스크립트를 dry-run 실행해 봅니다.
  ```powershell
  conda run -n env_isaaclab python scripts/play_ppo.py --num_envs 1 --checkpoint null
  ```
- 학습 루프가 정상적으로 가동되는지 검증하기 위해 train_ppo 스크립트를 10 iteration 정도 가동해 봅니다.
  ```powershell
  conda run -n env_isaaclab python scripts/train_ppo.py --num_envs 4 --max_iterations 10
  ```

### Manual Verification
- `play_ppo.py` 시연 환경에서 로봇 팔이 정육면체를 향해 움직이고 목표 위치(허리 앞 5cm)를 지시하는 마커 등이 잘 매칭되는지 뷰어를 통해 확인합니다.
