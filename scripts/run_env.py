# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
G1 + RUKA 커스텀 RL 환경을 실행하고 시각화하는 테스트 스크립트입니다.
이 스크립트는 시뮬레이터를 켜고 바닥, 조명, 로봇이 정상적으로 스폰되는지 확인합니다.

실행:
  .\isaaclab.bat -p C:\isaac\isaac-sim-project-with-ruka\scripts\run_env.py
"""

import argparse
import sys
import os

# 스크립트 실행 경로를 path에 추가하여 모듈을 임포트할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# 명령줄 인자 설정
parser = argparse.ArgumentParser(description="G1 + RUKA 커스텀 RL 환경 실행 스크립트")
parser.add_argument("--num_envs", type=int, default=4, help="스폰할 환경의 개수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim 앱 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""시뮬레이터 시작 후에 나머지 모듈을 로드합니다."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
from g1_ruka_env_cfg import G1RukaEnvCfg


def main():
    # 환경 설정 생성
    env_cfg = G1RukaEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # RL 환경 인스턴스 생성 (바닥, 조명, 로봇이 씬에 생성됨)
    print("[INFO]: G1 + RUKA 환경을 생성하는 중...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("[INFO]: 환경 생성 완료!")

    count = 0
    # 시뮬레이션 루프
    while simulation_app.is_running():
        with torch.inference_mode():
            # 일정 주기마다 리셋
            if count % 250 == 0:
                count = 0
                env.reset()
                print("-" * 60)
                print("[INFO]: 환경 리셋 수행됨...")

            # 로봇 관절의 수에 맞추어 제로 액션(Zero Action) 생성
            # action shape은 env.action_manager.action의 형태를 따릅니다.
            actions = torch.zeros_like(env.action_manager.action)

            # 환경 한 스텝 진행
            obs, rew, terminated, truncated, info = env.step(actions)

            # 디버깅 출력: 0번 환경의 일부 관측치 출력
            if count % 50 == 0:
                print(f"[Step {count:3d}]")
                # joint_pos의 첫 5개 값 출력
                print(f"  - 로봇 관절 상대 위치 (일부): {obs['policy'][0][:5].tolist()}")

            count += 1

    # 환경 종료
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
