"""
G1 왼팔(7) + RUKA 손(15) = 22개 관절 작동 테스트 (IsaacLab 2.1.0)

각 관절을 사인파로 가동범위 안에서 왕복시켜 정상 작동을 눈으로 확인한다.
허리(waist) 3개는 고정.

실행 (IsaacLab 폴더에서):
  .\isaaclab.bat -p C:\isaac\isaac-lab-capstone-project\scripts\test_joints.py
  옵션:
    --mode arm     팔 7개만 움직임
    --mode hand    손 15개만 움직임
    --mode all     22개 전부 (기본)
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str,
    default=r"C:\isaac\isaac-lab-capstone-project\combined\g1_left_arm_ruka_final.usd")
parser.add_argument("--mode", type=str, default="all", choices=["arm", "hand", "all"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext

# ---- 제어 대상 관절 이름 ----
ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
]
HAND_JOINTS = [
    "ruka_Index_MCP_Joint", "ruka_Index_DIP_Joint", "ruka_Index_PIP_Joint",
    "ruka_Middle_MCP_Joint", "ruka_Middle_DIP_Joint", "ruka_Middle_PIP_Joint",
    "ruka_Ring_MCP_Joint", "ruka_Ring_DIP_Joint", "ruka_Ring_PIP_Joint",
    "ruka_Pinky_MCP_Joint", "ruka_Pinky_DIP_Joint", "ruka_Pinky_PIP_Joint",
    "ruka_Thumb_CMC_Joint", "ruka_Thumb_MCP_Joint", "ruka_Thumb_IP_Joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.8, 0.8, 1.3], target=[0.2, 0.0, 0.9])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=3000.0))

    # ---- 로봇 + actuator 설정 ----
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        actuators={
            # 팔: 강한 PD (무거운 링크)
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["left_.*_joint"],
                stiffness=200.0, damping=20.0,
            ),
            # 손가락: 약한 PD (가벼운 링크)
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["ruka_.*"],
                stiffness=5.0, damping=0.5,
            ),
            # 허리: 고정 (강한 PD로 0 유지)
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"],
                stiffness=500.0, damping=50.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)
    sim.reset()

    # ---- 움직일 관절 인덱스 찾기 ----
    if args.mode == "arm":
        target_names = ARM_JOINTS
    elif args.mode == "hand":
        target_names = HAND_JOINTS
    else:
        target_names = ARM_JOINTS + HAND_JOINTS

    all_names = robot.joint_names
    print("\n" + "=" * 60)
    print(f"전체 관절 수: {robot.num_joints}  |  움직일 관절 수: {len(target_names)}  (mode={args.mode})")
    print("=" * 60)

    target_idx = []
    for n in target_names:
        if n in all_names:
            target_idx.append(all_names.index(n))
        else:
            print(f"  [경고] 관절 못 찾음: {n}")
    target_idx = torch.tensor(target_idx, device=args.device)

    # 각 관절의 가동범위(limit) 가져오기
    limits = robot.data.joint_pos_limits[0]  # (num_joints, 2)
    lo = limits[target_idx, 0]
    hi = limits[target_idx, 1]
    mid = (lo + hi) * 0.5
    amp = (hi - lo) * 0.4  # 가동범위의 80% 왕복

    # 기본 자세 = 현재값
    default_pos = robot.data.default_joint_pos.clone()

    print("움직이는 중... 뷰어를 확인하세요. (창 닫으면 종료)\n")
    t = 0.0
    while simulation_app.is_running():
        t += sim.get_physics_dt()
        # 목표 = 중앙 + 진폭*sin (관절마다 위상 약간 다르게)
        phase = torch.arange(len(target_idx), device=args.device) * 0.3
        targets = mid + amp * torch.sin(2.0 * math.pi * 0.2 * t + phase)

        joint_pos_target = default_pos.clone()
        joint_pos_target[0, target_idx] = targets

        robot.set_joint_position_target(joint_pos_target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
