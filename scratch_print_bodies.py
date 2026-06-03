import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", type=str, default=r"C:\isaac\isaac-sim-project-with-ruka\combined\g1_left_arm_ruka_final.usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    sim = SimulationContext(sim_cfg)
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
        ),
    )
    robot = Articulation(robot_cfg)
    sim.reset()
    print("=" * 60)
    print("BODY NAMES:")
    print(robot.body_names)
    print("=" * 60)

if __name__ == "__main__":
    main()
    simulation_app.close()
