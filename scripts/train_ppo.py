# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL on G1 + RUKA environment."""

import argparse
import sys
import os

# 스크립트 실행 경로를 path에 추가하여 모듈을 임포트할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# 명령줄 인자 설정
parser = argparse.ArgumentParser(description="Train PPO RL agent on G1 + RUKA environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--max_iterations", type=int, default=None, help="Number of training iterations.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import packages after launching simulation app
import gymnasium as gym
import torch
from datetime import datetime
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import custom configurations
from g1_ruka_env_cfg import G1RukaEnvCfg
from g1_ruka_ppo_cfg import G1RukaPPORunnerCfg


def main():
    # 1. 환경 및 에이전트 설정 인스턴스화
    env_cfg = G1RukaEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    agent_cfg = G1RukaPPORunnerCfg()
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # 2. 로그 디렉토리 설정
    log_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging training to: {log_dir}")

    # 3. Gymnasium 환경 생성
    env = gym.make("Isaac-G1-Ruka-v0", cfg=env_cfg)

    # 4. RSL-RL을 위해 환경 래핑
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 5. RSL-RL Runner 인스턴스화 및 학습 시작
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    
    # 학습 시작
    print("[INFO] Starting PPO training...")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # 6. 환경 종료
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
