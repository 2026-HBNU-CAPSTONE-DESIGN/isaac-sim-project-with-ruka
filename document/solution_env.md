# Isaac Lab 환경 (Windows)

## 환경

- Windows 11, Python 3.10, CUDA 12.1, RTX 3070
- Isaac Sim 4.5.0 (pip 설치)
- Isaac Lab 2.1.0

## 주의 사항 (시행착오로 알게 된 것)

- Long Path 활성화 필수 + 재부팅
- setuptools<81 사용 (pkg_resources 보존)
- NumPy<2 고정 (Isaac Sim 4.5 호환성)
- osqp 제거 필요 (Windows access violation)
  → wheeled_robots 사용 불가, 무시 OK
- h5py==3.10.0 (NumPy 1.x 호환)

## 설치 흐름

1. Long Path 켜고 재부팅
2. conda create -n env_isaaclab python=3.10
3. pip install "setuptools<81" wheel --force-reinstall
4. pip install torch==2.5.1 ... (CUDA 12.1)
5. pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
6. cd IsaacLab && isaaclab.bat --install
7. flatdict 에러 → --no-build-isolation으로 우회
8. pip install -e source\isaaclab
9. pip uninstall osqp qdldl -y
10. pip install "numpy<2" --force-reinstall --no-deps
11. pip install "h5py==3.10.0" --no-cache-dir --no-deps

## 검증

isaaclab.bat -p scripts\tutorials\00_sim\create_empty.py
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Cartpole-v0 --num_envs 16
