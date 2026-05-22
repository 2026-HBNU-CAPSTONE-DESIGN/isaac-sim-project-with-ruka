# G1 왼팔 + RUKA 왼손 결합 URDF

## 파일
- `g1_left_arm_ruka.urdf` — G1 29-DOF 전신 + RUKA 왼손 결합본 (55 link / 54 joint)
- `combine.py` — 재생성 스크립트 (roll 조정용)

## 결합 내용
- G1 더미 손(`left_hand_palm_joint` + `left_rubber_hand`) 제거
- RUKA 전체에 `ruka_` prefix (이름 충돌 방지)
- `left_wrist_yaw_link` → `ruka_base_link` fixed joint(`left_wrist_to_ruka`)
- attach origin: xyz="0.0415 0.003 0" (G1 기존 손 오프셋), rpy="0 1.5708 0" (손가락→+X 정렬)

## ⚠️ 사용 전 필수 작업

### 1. mesh 복사 (가장 먼저!)
URDF는 STL을 `meshes/` 경로로 참조. 두 로봇 메시를 모두 모아야 함.
```powershell
Copy-Item C:\isaac\g1_description\meshes\*.STL  C:\isaac\combined\meshes\
Copy-Item C:\isaac\RUKA\urdf\meshes_left\*.STL  C:\isaac\combined\meshes\
```

### 2. Isaac Sim 시각 검증 (roll 확정)
손가락 방향은 정렬됨. 손바닥 방향(roll)은 육안 확인 필요.
뒤집혔으면 combine.py의 `ROLL_DEG=180`, 옆으로 틀어졌으면 `90`/`-90`으로 바꿔 재실행.

### 3. IsaacLab 2.1.0 USD 변환
```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    /path/to/combined/g1_left_arm_ruka.urdf \
    /path/to/combined/g1_left_arm_ruka.usd \
    --merge-joints --fix-base
```
- `--fix-base`: pick&place에서 G1 상체 고정 (걸어다니지 않게)

## 제어 대상 joint (IsaacLab config용)
- 왼팔 7: left_shoulder_pitch/roll/yaw_joint, left_elbow_joint, left_wrist_roll/pitch/yaw_joint
- RUKA 15: ruka_{Index,Middle,Ring,Pinky}_{MCP,DIP,PIP}_Joint, ruka_Thumb_{CMC,MCP,IP}_Joint

## 구조 트리
pelvis → torso → left_shoulder(pitch/roll/yaw) → elbow → wrist(roll/pitch/yaw)
  →[fixed]→ ruka_base_link → ruka_Palm_Link → 5 fingers (각 MCP→DIP→PIP)
