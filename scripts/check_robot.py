"""
G1 + RUKA 왼손 USD 로드 테스트 스크립트 (IsaacLab 2.1.0)
- USD를 articulation으로 불러오고
- 관절 이름/개수를 출력하고
- 뷰어에 띄워 시뮬레이션을 돌려본다.

실행:
  cd C:\isaac\IsaacLab
  .\isaaclab.bat -p C:\isaac\isaac-lab-capstone-project\scripts\check_robot.py
"""

import argparse
from isaaclab.app import AppLauncher

# ---- CLI ----
parser = argparse.ArgumentParser(description="G1+RUKA USD 로드 테스트")
parser.add_argument(
    "--usd",
    type=str,
    default=r"C:\isaac\isaac-lab-capstone-project\combined\g1_left_arm_ruka_final.usd",
    help="로드할 USD 경로",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ---- Isaac Sim 앱 실행 ----
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- 이하 isaaclab import는 앱 실행 후에 ----
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext


def main():
    # 시뮬레이션 컨텍스트
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.5])

    # 바닥
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)

    # 조명
    light = sim_utils.DomeLightCfg(intensity=3000.0)
    light.func("/World/light", light)

    # ---- 로봇 articulation 설정 ----
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            # base를 공중에 고정 (pick&place는 걸어다니지 않음)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # 인접 link 자기충돌 끔 (손-손목 겹침 대응)
                fix_root_link=True,             # pelvis 고정
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),  # 공중에 띄움
        ),
        actuators={},  # 일단 비워둠 (로드 확인이 목적)
    )

    robot = Articulation(robot_cfg)

    # 시뮬레이션 리셋
    sim.reset()

    # ---- 관절 정보 출력 ----
    print("\n" + "=" * 60)
    print("로봇 로드 성공!")
    print("=" * 60)
    print(f"총 관절(DOF) 개수: {robot.num_joints}")
    print(f"총 body 개수: {robot.num_bodies}")
    print("\n--- 관절 이름 목록 ---")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")
    print("=" * 60 + "\n")

    # ---- 시뮬레이션 루프 (그냥 띄워서 보기) ----
    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
