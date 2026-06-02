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

-
