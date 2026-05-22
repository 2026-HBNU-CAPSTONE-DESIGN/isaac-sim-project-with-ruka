"""
RUKA URDF visual check in Isaac Lab.

What this does:
  1. Launches Isaac Sim in GUI mode.
  2. Converts the URDF to USD on the fly using Isaac Lab's UrdfConverter.
  3. Spawns the hand fixed in mid-air at the origin.
  4. Sweeps each finger joint with a slow sine wave so you can SEE
     whether the hand bends toward the palm (good) or the back (bad).
  5. Prints diagnostics: link names, joint names, joint limits, dof order.

Run with the Isaac Lab Python:
    cd C:\\path\\to\\IsaacLab
    .\\isaaclab.bat -p C:\\isaac\\RUKA\\urdf\\visual_check.py

Or, if you have a working `python` from the isaaclab conda env:
    python C:\\isaac\\RUKA\\urdf\\visual_check.py

Edit URDF_PATH below if your URDF lives elsewhere.
"""

import argparse
import os

# ---- Isaac Lab AppLauncher must run BEFORE any other isaaclab/omni imports.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--urdf", type=str,
                    default=r"C:\isaac\RUKA\urdf\ruka_left.urdf",
                    help="Path to the RUKA URDF.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Now safe to import the rest.
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.actuators import ImplicitActuatorCfg


def convert_urdf_to_usd(urdf_path: str) -> str:
    """Use Isaac Lab's UrdfConverter to produce a USD next to the URDF."""
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(urdf_path)

    out_dir = os.path.dirname(os.path.abspath(urdf_path))
    usd_dir = os.path.join(out_dir, "usd")
    os.makedirs(usd_dir, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        usd_file_name="ruka_left.usd",
        force_usd_conversion=True,           # always re-convert during dev
        fix_base=True,                       # palm fixed in mid-air
        merge_fixed_joints=False,
        # Default joint drive: position control with reasonable stiffness
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=20.0, damping=1.0,
            ),
        ),
        # Self collisions OFF for first-time check; many false contacts
        # otherwise mask real geometry problems.
        collider_type="convex_decomposition",
    )
    converter = UrdfConverter(cfg)
    print(f"[convert] USD written to: {converter.usd_path}")
    return converter.usd_path


def main():
    usd_path = convert_urdf_to_usd(args.urdf)

    # ---- Build a simple scene
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.25, 0.25, 0.25), target=(0.0, 0.0, 0.1))

    # Ground plane for orientation reference
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # Bright dome light (intensity must be passed via the SAME cfg used in func())
    dome_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(1.0, 1.0, 1.0),
    )
    dome_cfg.func("/World/DomeLight", dome_cfg)

    # Add a directional light from above-front so the hand has clear shading
    distant_cfg = sim_utils.DistantLightCfg(
        intensity=2000.0,
        color=(1.0, 1.0, 1.0),
        angle=0.53,
    )
    distant_cfg.func("/World/DistantLight", distant_cfg,
                     translation=(0.0, 0.0, 1.0),
                     orientation=(0.7071, 0.0, 0.7071, 0.0))

    # ---- ArticulationCfg pointing at our converted USD
    ruka_cfg = ArticulationCfg(
        prim_path="/World/RUKA",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,        # study geometry, not dynamics
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            # Identity quat (w, x, y, z); palm at origin facing as URDF defines.
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=20.0,
                damping=1.0,
                effort_limit=10.0,
                velocity_limit=10.0,
            ),
        },
    )

    ruka = Articulation(ruka_cfg)

    sim.reset()

    # ---- Diagnostics
    print("\n=== ARTICULATION DIAGNOSTICS ===")
    print(f"Body names ({len(ruka.body_names)}):")
    for n in ruka.body_names:
        print(f"  {n}")
    print(f"\nJoint names ({len(ruka.joint_names)}):")
    for i, n in enumerate(ruka.joint_names):
        lo = ruka.data.joint_limits[0, i, 0].item()
        hi = ruka.data.joint_limits[0, i, 1].item()
        print(f"  [{i:2d}] {n:25s}  range=[{lo:+.3f}, {hi:+.3f}] rad")

    # ---- Sweep joints with a slow sine
    # We sweep MCP, PIP, DIP joints of all four fingers together so we
    # can see them flex toward the palm. Thumb has its own range.
    finger_joint_names = [n for n in ruka.joint_names
                          if any(k in n for k in
                                 ["Index", "Middle", "Ring", "Pinky"])]
    thumb_joint_names = [n for n in ruka.joint_names if "Thumb" in n]
    finger_idx = [ruka.joint_names.index(n) for n in finger_joint_names]
    thumb_idx = [ruka.joint_names.index(n) for n in thumb_joint_names]

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    print("\nStarting sweep. Watch the GUI.")
    print("  Fingers (Index/Middle/Ring/Pinky) flex 0 -> ~1.5 rad and back.")
    print("  Thumb_CMC sweeps its negative range; Thumb_MCP/IP flex.")
    print("  If a joint bends AWAY from the palm, its axis is flipped.\n")

    target_pos = torch.zeros_like(ruka.data.joint_pos)

    while simulation_app.is_running():
        # Wave between 0 and 1: phase = (1 - cos)/2
        phase = 0.5 * (1.0 - math.cos(0.5 * sim_time))   # 0 .. 1, slow

        # All joints: sweep from one limit to the other.
        for i in finger_idx + thumb_idx:
            lo = ruka.data.joint_limits[0, i, 0].item()
            hi = ruka.data.joint_limits[0, i, 1].item()
            # Pick a target endpoint that is the non-zero limit (so we
            # actually move the joint). For limits like [-2.44, 0] we
            # sweep 0 -> -2.44 (negative direction).
            if abs(lo) > abs(hi):
                target = lo
            else:
                target = hi
            # Cap amplitude at 1.5 rad for visibility regardless of sign.
            target = max(-1.5, min(1.5, target))
            target_pos[0, i] = target * phase

        ruka.set_joint_position_target(target_pos)
        ruka.write_data_to_sim()

        sim.step()
        ruka.update(sim_dt)
        sim_time += sim_dt


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
