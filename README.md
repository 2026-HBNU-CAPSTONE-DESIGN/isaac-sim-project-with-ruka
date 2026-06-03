# isaac-lab-capstone-project

## 스펙

Hand: RUKA-Hand v1
Arm: Unitree G1

## 1. 로컬 환경 Isaac Sim/Lab 설치 (pip을 활용한 설치)

> 먼저 알아둘 점: Isaac Sim 4.5 ↔ Isaac Lab 2.0.x ~ 2.1.x
> 버전 매칭을 해야 한다. Isaac Lab 2.3+ 버전은 Isaac Sim 5.x 버전을 요구함. (고사양 PC 요구)

```bash
# 1) Conda로 Python 3.10 환경 생성
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab

# 2) PyTorch 설치 (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 3) Isaac Sim 4.5 pip 설치
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# 4) Isaac Lab 소스 클론
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.1.0   # Isaac Sim 4.5와 호환되는 태그

# 5) Isaac Lab 설치 (모든 RL 프레임워크 포함)
# ./isaaclab.sh --install      # Linux
isaaclab.bat --install     # Windows
```

- 충돌 확인 : `pip check`

```bash
conda activate env_isaaclab

:: osqp와 의존성 모두 제거
pip uninstall osqp qdldl -y

:: NumPy가 다시 2.x로 올라갔다면 1.x로 복원
pip install "numpy<2" --force-reinstall

:: 확인
python -c "import numpy; print(numpy.__version__)"

:: 시뮬 다시 시도
cd C:\isaac\IsaacLab
isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
```

## 2. Isaac Sim 상에서 팔과 손 작동 확인

- conda 가상환경 활성화시킨 후 실행해야 함.

```powershell
# 팔만 움직이기
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\test_joints.py --mode arm

# 손만 움직이기
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\test_joints.py --mode hand

# 손과 팔 움직이기(22관절 손+팔)
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\test_joints.py --mode all
```

## 3. G1 + RUKA 커스텀 학습 환경 실행 (기초 설정 - 바닥 및 로봇 스폰)

커스텀 매니저 기반(Manager-based) RL 환경 설정을 실행하고 시각적으로 확인합니다. 이 환경은 바닥(Ground plane), 조명(Dome light), G1 + RUKA 로봇을 포함합니다.

```powershell
# 가상환경 활성화 후 실행
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\run_env.py
```

## 4. 강화학습 (PPO) 학습 및 최종 모델 확인

이 프로젝트는 `rsl_rl` 기반의 PPO 알고리즘을 사용해 G1 + RUKA 로봇의 목표물 도달(Reach) 작업을 학습시킵니다.

### 4.1. 강화학습 시작 (Train PPO)
가상환경(`env_isaaclab`)이 활성화된 터미널에서 다음 명령어를 실행합니다.

```powershell
# 1) 헤드리스(Headless) 모드로 학습 진행 (속도가 빠르며 리소스를 적게 소모합니다, 기본 환경 수: 16)
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\train_ppo.py --num_envs 16 --headless

# 2) GUI를 띄운 상태로 실시간 관찰하며 학습 진행
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\train_ppo.py --num_envs 8

# (옵션) 최대 학습 반복 횟수(iteration) 지정 (기본값: 1000)
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\train_ppo.py --num_envs 16 --max_iterations 500 --headless
```

### 4.2. 학습 결과 및 최종 모델 확인 (Play Trained Policy)
학습 진행 중 혹은 완료 후 저장된 가중치 체크포인트(`.pt` 파일)를 로드하여 시각적으로 동작 성능을 확인합니다.

```powershell
# 1) 가장 최근에 진행한 학습 세션의 마지막 체크포인트를 자동으로 찾아 로드하여 시동
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\play_ppo.py --num_envs 1

# 2) 특정 학습 결과의 특정 체크포인트를 지정하여 실행
.\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\play_ppo.py --num_envs 1 --checkpoint "C:\isaac\isaac-sim-project-with-ruka\logs\rsl_rl\g1_ruka_reach\YYYY-MM-DD_HH-MM-SS\model_X.pt"
```
* **로그 및 체크포인트 경로**: 학습 기록 및 파라미터는 `C:\isaac\isaac-sim-project-with-ruka\logs\rsl_rl\g1_ruka_reach\` 하위 디렉토리에 타임스탬프별로 저장됩니다.

