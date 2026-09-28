# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


from . import mdp
from .reach_cfg import ReachSceneCfg, ReachEnvCfg, EventCfg

##
# Scene definition
##

@configclass
class ReachRGBCameraSceneCfg(ReachSceneCfg):

    # add camera to the scene
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(2.0, 0.0, 0.4), rot=(-0.5, 0.5, 0.5, -0.5), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        width=84,
        height=84,
    )

    # target point visualizer
    # target_viz: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/TargetViz",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.05,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         # rigid_props=None,  # 동역학 끔(장식용). 필요하면 kinematic_enabled=True 등으로 조절
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.3, 0.0, 0.15),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #     ),
    # )

    target_viz: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetViz",
        spawn=sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.3, 0.0, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class EventVizCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    sync_target_viz_to_ee_pose = EventTerm(
        func=mdp.set_prim_pose_from_command,  # ← 아래 2-3의 헬퍼 함수
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "command_name": "ee_pose",
        },
    )


@configclass
class EventVizCNNCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    sync_target_viz_to_ee_pose = EventTerm(
        func=mdp.set_prim_pose_from_command,  # ← 아래 2-3의 헬퍼 함수
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "command_name": "ee_pose",
            "camera_pos_raw": (2.0, 0.0, 0.4)
        },
    )

# @configclass
# class RGBObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class RGBCameraPolicyCfg(ObsGroup):
#         """Observations for policy group with RGB images."""

#         image = ObsTerm(func=mdp.image_features,
#                         params={"sensor_cfg": SceneEntityCfg("tiled_camera"),
#                                 "data_type": "rgb",
#                                 "model_name": "resnet18"})
#         pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
#         actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     policy: ObsGroup = RGBCameraPolicyCfg()


# @configclass
# class RGBObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class RGBCameraPolicyCfg(ObsGroup):
#         """Observations for policy group with RGB images."""

#         image = ObsTerm(func=mdp.image_features,
#                         params={"sensor_cfg": SceneEntityCfg("tiled_camera"),
#                                 "data_type": "rgb",
#                                 "model_name": "resnet18"})
#         # actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class RGBObservationsFeatureCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        image = ObsTerm(func=mdp.image_features,
                        params={"sensor_cfg": SceneEntityCfg("tiled_camera"),
                                "data_type": "rgb",
                                "model_name": "resnet18"})
        # actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class RGBObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):

        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), 
                                                "data_type": "rgb",
                                                "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


@configclass
class CNNObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):

        image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), 
                                                "data_type": "rgb"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: ObsGroup = RGBCameraPolicyCfg()


# @configclass
# class RGBObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class RGBCameraPolicyCfg(ObsGroup):

#         image = ObsTerm(func=mdp.image_stack, params={"sensor_cfg": SceneEntityCfg("tiled_camera"),
#                                                       "data_type": "rgb"})

#         def __post_init__(self):
#             self.enable_corruption = False
#             self.concatenate_terms = True

#     policy: ObsGroup = RGBCameraPolicyCfg()



@configclass
class ResNet18ObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ResNet18FeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen ResNet18."""

        image = ObsTerm(
            func=mdp.image_features,
            params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb", "model_name": "resnet18"},
        )

    policy: ObsGroup = ResNet18FeaturesCameraPolicyCfg()


@configclass
class TheiaTinyObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class TheiaTinyFeaturesCameraPolicyCfg(ObsGroup):
        """Observations for policy group with features extracted from RGB images with a frozen Theia-Tiny Transformer"""

        image = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("tiled_camera"),
                "data_type": "rgb",
                "model_name": "theia-tiny-patch16-224-cddsv",
                "model_device": "cuda:0",
            },
        )

    policy: ObsGroup = TheiaTinyFeaturesCameraPolicyCfg()


@configclass
class ReachRGBEnvCfg(ReachEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachRGBCameraSceneCfg = ReachRGBCameraSceneCfg(num_envs=128, env_spacing=10)
    observations: RGBObservationsCfg = RGBObservationsCfg()
    events: EventVizCfg = EventVizCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)


@configclass
class ReachCNNEnvCfg(ReachEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachRGBCameraSceneCfg = ReachRGBCameraSceneCfg(num_envs=128, env_spacing=10)
    observations: RGBObservationsCfg = RGBObservationsCfg()
    events: EventVizCNNCfg = EventVizCNNCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)


@configclass
class ReachRGBFeatureEnvCfg(ReachEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachRGBCameraSceneCfg = ReachRGBCameraSceneCfg(num_envs=128, env_spacing=10)
    observations: RGBObservationsFeatureCfg = RGBObservationsFeatureCfg()
    events: EventVizCfg = EventVizCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # remove ground as it obstructs the camera
        self.scene.ground = None
        # viewer settings
        self.viewer.eye = (7.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 2.5)

