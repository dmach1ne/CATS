"""Table geometry with an extension toward the robot base."""
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from .vertical15_fixedcube_env_cfg import Vertical15FixedCubeEnvCfg, Vertical15FixedCubeEvalEnvCfg

# The base edge and cutout dimensions are simulation geometry estimates.
BASE_FRONT_X = 0.07156231
MAIN_TABLE_NEAR_X = BASE_FRONT_X + 0.125
TABLE_BOUNDS_X = (MAIN_TABLE_NEAR_X, MAIN_TABLE_NEAR_X + 0.80)
EXTENSION_LENGTH = 0.55
EXTENSION_BACK_X = MAIN_TABLE_NEAR_X - EXTENSION_LENGTH
# Camera image-right points toward +world Y in the recorded policy camera.
EXTENSION_Y = (-0.60, 0.60 - 0.125)
OPENING_X = (-0.16, 0.08)
OPENING_Y = (-0.10, 0.10)
TABLE_TOP_Z = 0.021
BOARD_THICKNESS = 0.005
WORK_SURFACE_Z = TABLE_TOP_Z + BOARD_THICKNESS
TABLE_THICKNESS = 0.03  # Assumed table thickness.


@configclass
class MatteCubeMaterialCfg(sim_utils.PreviewSurfaceCfg):
    # Candidate C: decode sRGB #408F52 to linear RGB, then multiply by 0.75.
    diffuse_color = (0.03845209378053243, 0.20600798404528847, 0.06328215865811161)
    roughness = 1.0
    metallic = 0.0
    use_specular_workflow: int = 1
    specular_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ior: float = 1.0
    clearcoat: float = 0.0


class _Table55Mixin:
    def _configure_table55(self):
        def slab(name, bounds, thickness, center_z):
            x0, x1, y0, y1 = bounds
            return AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=(x1-x0, y1-y0, thickness),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1., 1., 1.)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=((x0+x1)/2, (y0+y1)/2, center_z)),
            )
        x0, x1 = TABLE_BOUNDS_X
        hx0, hx1 = OPENING_X
        hy0, hy1 = OPENING_Y
        ey0, ey1 = EXTENSION_Y
        pieces = ((x0, x1, -.6, .6), (hx1, x0, ey0, ey1),
                  (EXTENSION_BACK_X, hx0, ey0, ey1),
                  (hx0, hx1, ey0, hy0), (hx0, hx1, hy1, ey1))
        for prefix, thickness, z in (("table", TABLE_THICKNESS, TABLE_TOP_Z-TABLE_THICKNESS/2),
                                      ("protective_board", BOARD_THICKNESS, TABLE_TOP_Z+BOARD_THICKNESS/2)):
            for index, bounds in enumerate(pieces):
                name = prefix if index == 0 else f"{prefix}_{index}"
                setattr(self.scene, name, slab(name, bounds, thickness, z))
        self.scene.object.spawn.visual_material = MatteCubeMaterialCfg()
        self.scene.object.init_state.pos = (0.45, 0., WORK_SURFACE_Z + 0.049 / 2)
        goal_z = WORK_SURFACE_Z + 0.049 / 2 + 0.15 - self.scene.robot.init_state.pos[2]
        self.commands.object_pose.ranges.pos_z = (goal_z, goal_z)
        self.rewards.lifting_object.params["minimum_height"] = WORK_SURFACE_Z + 0.049 / 2
        self.rewards.lifting_object.params["maximum_height"] = WORK_SURFACE_Z + 0.10
        for name in ("object_goal_tracking", "object_goal_tracking_fine_grained"):
            term = getattr(self.rewards, name)
            term.params["minimal_height"] += WORK_SURFACE_Z


@configclass
class Vertical15Table55EnvCfg(_Table55Mixin, Vertical15FixedCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_table55()


@configclass
class Vertical15Table55EvalEnvCfg(_Table55Mixin, Vertical15FixedCubeEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_table55()
