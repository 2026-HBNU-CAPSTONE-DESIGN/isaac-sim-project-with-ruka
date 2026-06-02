# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.envs.mdp as mdp


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
                stiffness=5.0,
                damping=0.5,
            ),
            # 허리 조인트: 고정 (강한 PD 제어)
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*"],
                stiffness=500.0,
                damping=50.0,
            ),
        },
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 기본 더미 보상 설정 (시뮬레이터 로드용)
    dummy_reward = RewTerm(
        func=lambda env: 0.0,
        weight=0.0,
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
        self.episode_length_s = 5.0
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.3)
        self.sim.dt = 1.0 / 60.0
