# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.envs.mdp as mdp
import torch
import gymnasium as gym
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from g1_ruka_ppo_cfg import G1RukaPPORunnerCfg


def object_hand_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
) -> torch.Tensor:
    """Reward for moving the hand close to the target object using tanh kernel."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Resolve body_ids dynamically if not already resolved by the manager
    if isinstance(robot_cfg.body_ids, slice):
        body_ids, _ = robot.find_bodies(robot_cfg.body_names)
    else:
        body_ids = robot_cfg.body_ids

    # target object position: (num_envs, 3)
    object_pos = object.data.root_pos_w
    # palm link position: (num_envs, 3)
    hand_pos = robot.data.body_pos_w[:, body_ids[0]]
    # distance: (num_envs,)
    dist = torch.norm(object_pos - hand_pos, dim=1)
    return 1.0 - torch.tanh(dist / std)


def hand_position_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
) -> torch.Tensor:
    """World position of the hand (left wrist yaw link)."""
    robot: Articulation = env.scene[robot_cfg.name]
    if isinstance(robot_cfg.body_ids, slice):
        body_ids, _ = robot.find_bodies(robot_cfg.body_names)
    else:
        body_ids = robot_cfg.body_ids
    return robot.data.body_pos_w[:, body_ids[0]]


def object_relative_to_hand(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"]),
) -> torch.Tensor:
    """Relative position of the object with respect to the hand: object_pos - hand_pos."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    if isinstance(robot_cfg.body_ids, slice):
        body_ids, _ = robot.find_bodies(robot_cfg.body_names)
    else:
        body_ids = robot_cfg.body_ids

    object_pos = object.data.root_pos_w
    hand_pos = robot.data.body_pos_w[:, body_ids[0]]
    return object_pos - hand_pos


def joint_deviation_penalty(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation of specific joint positions from their default values."""
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Find joint indices
    joint_ids, _ = robot.find_joints(joint_names)
    
    # Current and default positions
    curr_pos = robot.data.joint_pos[:, joint_ids]
    default_pos = robot.data.default_joint_pos[:, joint_ids]
    
    # L2 deviation: (num_envs,)
    return torch.sum(torch.square(curr_pos - default_pos), dim=1)


def multi_stage_manipulation_rewards(
    env: ManagerBasedRLEnv,
    key: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Multi-stage rewards for reaching, lifting, and placing the object."""
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Find body ids
    palm_ids, _ = robot.find_bodies("left_wrist_yaw_link")
    pelvis_ids, _ = robot.find_bodies("pelvis")

    # Positions
    palm_pos = robot.data.body_pos_w[:, palm_ids[0]]
    pelvis_pos = robot.data.body_pos_w[:, pelvis_ids[0]]
    object_pos = object.data.root_pos_w

    # Place target is 5cm in front of the pelvis, and 15cm above it
    target_place_pos = pelvis_pos + torch.tensor([0.05, 0.0, 0.15], device=env.device)

    # Distances
    dist_hand_obj = torch.norm(object_pos - palm_pos, dim=1)
    dist_obj_target = torch.norm(object_pos - target_place_pos, dim=1)

    # Stages
    is_reached = (dist_hand_obj < 0.10).float()
    
    # Initial cube height is 0.0375
    lift_height = object_pos[:, 2] - 0.0375
    lift_height = torch.clamp(lift_height, min=0.0)
    is_lifted = (lift_height > 0.05).float()

    if key == "reach":
        return 1.0 - torch.tanh(dist_hand_obj / 0.25)
    elif key == "lift":
        return is_reached * torch.clamp(lift_height * 10.0, max=1.0)
    elif key == "place":
        return is_lifted * (1.0 - torch.tanh(dist_obj_target / 0.15))
    else:
        return torch.zeros_like(dist_hand_obj)


@configclass
class G1RukaSceneCfg(InteractiveSceneCfg):
    """Configuration for the G1 + RUKA scene."""

    # ground plane (바닥)
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # lights (조명)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    # robot (G1 왼팔 + RUKA 왼손)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=r"C:\isaac\isaac-sim-project-with-ruka\combined\g1_left_arm_ruka_final.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
        actuators={
            # G1 왼팔 조인트: 강한 PD 제어 (무거운 링크)
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["left_.*_joint"],
                stiffness=200.0,
                damping=20.0,
            ),
            # RUKA 손가락 조인트: 약한 PD 제어 (가벼운 링크)
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["ruka_.*"],
                stiffness=2.0,
                damping=0.2,
            ),
            # 허리 조인트: 고정 (강한 PD 제어)
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"],
                stiffness=500.0,
                damping=50.0,
            ),
        },
    )

    # target object (정육면체 큐브 - 크기 7.5cm, 질량 100g)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.075, 0.075, 0.075),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.24, 0.12, 0.0375),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # G1 왼팔 관절 제어 설정 (PD Target)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_.*_joint"],
        scale=0.5,
        use_default_offset=True,
    )
    # RUKA 손가락 관절 제어 설정 (PD Target)
    hand_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["ruka_.*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        hand_pos = ObsTerm(func=hand_position_w)
        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rel_hand = ObsTerm(func=object_relative_to_hand)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.20, 0.28), "y": (0.08, 0.16), "z": (0.0375, 0.0375)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # object_hand_distance = RewTerm(
    #     func=object_hand_distance,
    #     weight=10.0,
    #     params={"std": 0.2, "robot_cfg": SceneEntityCfg("robot", body_names=["left_wrist_yaw_link"])},
    # )

    reward_reach = RewTerm(
        func=multi_stage_manipulation_rewards,
        weight=15.0,
        params={"key": "reach"},
    )

    reward_lift = RewTerm(
        func=multi_stage_manipulation_rewards,
        weight=15.0,
        params={"key": "lift"},
    )

    reward_place = RewTerm(
        func=multi_stage_manipulation_rewards,
        weight=20.0,
        params={"key": "place"},
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
    )

    wrist_penalty = RewTerm(
        func=joint_deviation_penalty,
        weight=-0.5,
        params={
            "joint_names": [
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 에피소드 최대 길이 도달 시 종료
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class G1RukaEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 + RUKA environment."""

    # Scene settings
    scene: G1RukaSceneCfg = G1RukaSceneCfg(num_envs=4, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.sim.dt = 1.0 / 60.0


# Register Gym environment
gym.register(
    id="Isaac-G1-Ruka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "cfg_entry_point": G1RukaEnvCfg,
        "rsl_rl_cfg_entry_point": G1RukaPPORunnerCfg,
    },
)
