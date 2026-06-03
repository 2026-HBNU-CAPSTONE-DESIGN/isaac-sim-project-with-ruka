# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of G1 + RUKA RL agent trained with RSL-RL."""

import argparse
import sys
import os
import glob

# 스크립트 실행 경로를 path에 추가하여 모듈을 임포트할 수 있도록 합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# 명령줄 인자 설정
parser = argparse.ArgumentParser(description="Play trained PPO RL agent on G1 + RUKA environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint file.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import packages after launching simulation app
import gymnasium as gym
import torch
import time
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

    # 2. 체크포인트 경로 탐색
    resume_path = args_cli.checkpoint
    if resume_path is None:
        # 최신 학습 폴더와 체크포인트를 자동으로 검색
        log_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        
        # 최신 디렉토리 검색
        run_dirs = sorted(glob.glob(os.path.join(log_root_path, "202*")))
        if not run_dirs:
            print(f"[ERROR] No training logs found in: {log_root_path}")
            return
            
        latest_run_dir = run_dirs[-1]
        # model_*.pt 파일들 중 가장 최근 것 검색
        model_paths = sorted(glob.glob(os.path.join(latest_run_dir, "model_*.pt")), key=os.path.getmtime)
        if not model_paths:
            print(f"[ERROR] No checkpoints (*.pt) found in: {latest_run_dir}")
            return
            
        resume_path = model_paths[-1]
        
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # 3. Gymnasium 환경 생성
    env = gym.make("Isaac-G1-Ruka-v0", cfg=env_cfg)

    # 4. RSL-RL을 위해 환경 래핑
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 5. RSL-RL Runner 인스턴스화 및 로드
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # 학습된 정책(Policy) 가져오기
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 6. 추론 루프 실행
    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    
    print("[INFO] Running policy. Close viewer to exit.")
    while simulation_app.is_running():
        start_time = time.time()
        
        with torch.inference_mode():
            # 관측치(obs)로부터 행동(actions) 예측
            actions = policy(obs)
            # 환경 한 스텝 진행
            obs, _, _, _ = env.step(actions)
            
        # Real-time 동기화
        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 환경 종료
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
