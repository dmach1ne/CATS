"""학습된 정책을 목표 N회 초기화 동안 굴려 성공률을 재고 영상으로 남긴다.

성공 판정이 태스크에 내장돼 있지 않다(종료조건은 time_out 뿐, 보상은 거리 shaping).
그래서 여기서 명시적으로 정의한다.

    한 목표에 대해, 그 목표가 유효한 6초(=180 제어스텝) 동안
    panda_hand 와 목표점의 3D 거리 최솟값이 임계값 미만이면 성공.

임계값은 하나로 고정하지 않고 여러 개를 동시에 보고한다. 보상의 tanh 커널이
std=0.1 이라 10cm 에서 이미 보너스가 24% 로 떨어지고 거리벌점과 상쇄되므로,
'성공' 을 10cm 로 잡으면 지나치게 관대하다. 2~5cm 가 실질적인 구간이다.

주의: 기준 프레임은 panda_hand(손목)이다. 손끝은 그보다 5.84cm 앞에 있다.

사용:
    python scripts/eval_success.py --checkpoint <pth> --targets 8 --out <mp4>
    # 기본값은 평가 seed 12345, 22345, 32345를 각각 실행하고 통합한다.
    # 단일 seed가 필요하면 --seed 12345를 명시한다.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Franka-Reach-Eval-v0")
parser.add_argument("--targets", type=int, default=8, help="평가할 목표 초기화 횟수")
parser.add_argument("--out", type=str, default="eval.mp4")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--video-layout",
    choices=("camera", "high_res_dual", "policy_input_high_res"),
    default="policy_input_high_res",
)
parser.add_argument("--control-trace", action="store_true",
                    help="Append an action trace to the camera layout. The default layout already includes one.")
parser.add_argument("--seed", type=int, default=None, help="Single evaluation seed")
parser.add_argument(
    "--eval-seeds", type=int, nargs="+", default=None,
    help="Evaluation seeds to run sequentially; defaults to 12345 22345 32345",
)
parser.add_argument("--label", type=str, default="")
parser.add_argument("--agent-cfg", type=str, required=True,
                    help="Path to the rl_games agent YAML; defaults to the historical project path.")
parser.add_argument("--summary", type=str, default=None,
                    help="Optional JSON path for structured evaluation metrics.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

DEFAULT_EVAL_SEEDS = (12345, 22345, 32345)
THRESHOLDS = (0.02, 0.03, 0.05, 0.10, 0.15)
HOLD_THRESHOLDS = (0.02, 0.05, 0.10)
HOLD_DURATION_SEC = 1.0
POLICY_UP = 6
REACH_ACTION_LABELS = ("J1", "J2", "J3", "J4", "J5", "J6", "J7")
CFG_YAML = None  # --agent-cfg is required.


def _seed_path(path: str, seed: int) -> str:
    value = Path(path)
    return str(value.with_name(f"{value.stem}.seed{seed}{value.suffix}"))


def _aggregate_reach_summaries(reports: list[dict], seeds: list[int]) -> dict:
    """Pool target-level arrays while retaining each seed's summary."""
    import numpy as np

    first = reports[0]
    result = {
        "task": first["task"],
        "checkpoint": first["checkpoint"],
        "agent_cfg": first["agent_cfg"],
        "seed": None,
        "evaluation_seeds": seeds,
        "num_targets": sum(report["num_targets"] for report in reports),
        "per_seed_summaries": [
            {"seed": seed, "summary": report} for seed, report in zip(seeds, reports)
        ],
    }

    distances = np.concatenate(
        [np.asarray(report["min_distance_m"]["per_target"], dtype=np.float64) for report in reports]
    )
    result["min_distance_m"] = {
        "per_target": distances.tolist(),
        "median": float(np.median(distances)),
        "mean": float(distances.mean()),
        "max": float(distances.max()),
    }
    result["minimum_distance_success_rate"] = {
        f"lt_{int(threshold * 100)}cm": float((distances < threshold).mean())
        for threshold in THRESHOLDS
    }
    hold_reports = [report.get("hold_success_1s") for report in reports]
    if all(isinstance(report, dict) for report in hold_reports):
        result["hold_success_1s"] = {}
        for threshold in HOLD_THRESHOLDS:
            key = f"lt_{int(threshold * 100)}cm"
            values = np.concatenate([
                np.asarray(report[key]["per_target"], dtype=bool)
                for report in hold_reports
            ])
            result["hold_success_1s"][key] = {
                "required_duration_s": HOLD_DURATION_SEC,
                "per_target": values.tolist(),
                "rate": float(values.mean()),
            }
        result["success_rate"] = {
            key: value["rate"]
            for key, value in result["hold_success_1s"].items()
        }
    trace_reports = [report.get("distance_trace_m") for report in reports]
    if all(isinstance(report, dict) for report in trace_reports):
        result["distance_trace_m"] = {
            "control_dt": float(trace_reports[0]["control_dt"]),
            "per_target": [
                trace
                for report in trace_reports
                for trace in report["per_target"]
            ],
        }
    for name in ("approx_arm_energy_j", "arm_smoothness", "approx_arm_mean_power_w"):
        values = np.concatenate(
            [np.asarray(report[name]["per_target"], dtype=np.float64) for report in reports]
        )
        result[name] = {
            "per_target": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }

    action_reports = [report.get("policy_actions") for report in reports]
    if all(action_reports):
        result["policy_actions"] = {
            "definition": action_reports[0]["definition"],
            "control_dt": action_reports[0]["control_dt"],
            "values": [
                action
                for action_report in action_reports
                for action in action_report["values"]
            ],
        }

    slopes = [report["target_following"] for report in reports if report.get("target_following")]
    if slopes:
        result["target_following"] = {
            "mean": float(np.mean([item["mean"] for item in slopes])),
            "per_axis": np.mean([item["per_axis"] for item in slopes], axis=0).tolist(),
            "r2_mean": float(np.mean([item["r2_mean"] for item in slopes])),
        }
    else:
        result["target_following"] = None
    result["metric_definition"] = first["metric_definition"]
    return result


def _run_eval_seed_sweep() -> None:
    seeds = [int(seed) for seed in args.eval_seeds]
    summary_path = args.summary or str(Path(args.out).with_suffix(".summary.json"))
    reports = []
    for index, seed in enumerate(seeds):
        child_out = args.out if index == 0 else _seed_path(args.out, seed)
        child_summary = _seed_path(summary_path, seed)
        command = [
            sys.executable, os.path.abspath(__file__),
            "--checkpoint", args.checkpoint,
            "--task", args.task,
            "--targets", str(args.targets),
            "--out", child_out,
            "--fps", str(args.fps),
            "--video-layout", args.video_layout,
            "--seed", str(seed),
            "--summary", child_summary,
            "--headless", "--enable_cameras",
        ]
        if args.label:
            command.extend(["--label", f"{args.label} seed={seed}"])
        if args.agent_cfg:
            command.extend(["--agent-cfg", args.agent_cfg])
        if args.control_trace:
            command.append("--control-trace")
        subprocess.run(command, check=True)
        with open(child_summary, encoding="utf-8") as stream:
            reports.append(json.load(stream))

    aggregate = _aggregate_reach_summaries(reports, seeds)
    output = Path(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation_seeds": seeds, **aggregate["success_rate"]}, indent=2))
    print(f">>> MULTI-SEED SUMMARY WRITTEN {output}", flush=True)


if args.eval_seeds is None and args.seed is None:
    args.eval_seeds = list(DEFAULT_EVAL_SEEDS)
if args.eval_seeds is not None and args.seed is not None:
    parser.error("use either --seed or --eval-seeds, not both")
if args.eval_seeds is not None:
    _run_eval_seed_sweep()
    raise SystemExit(0)
args.seed = DEFAULT_EVAL_SEEDS[0] if args.seed is None else args.seed
args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import er_isaac.tasks  # noqa: F401, E402
from models.visual_proprio_network import register_visual_proprio_networks  # noqa: E402
from evaluation.high_res_video import (  # noqa: E402
    ACTION_TRACE_HEIGHT,
    DUAL_WIDTH,
    FULL_CAMERA_ORIGIN,
    FULL_HEIGHT,
    FULL_WIDTH,
    GUTTER,
    INPUT_ROW_HEIGHT,
    POLICY_VIEW_SIDE,
    STANDARD_VIDEO_WIDTH,
    compose_action_trace,
    compose_high_res_dual,
    compose_policy_input_high_res,
    full_to_policy_pixel,
    latest_policy_rgb,
)
from evaluation.action_metrics import mechanical_energy, arm_smoothness  # noqa: E402
from evaluation.hold_metrics import hold_success  # noqa: E402


register_visual_proprio_networks()

def quat_to_R(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def project_world(cam, p_w):
    t = cam.data.pos_w[0].detach().cpu().numpy().astype(np.float64)
    R = quat_to_R(cam.data.quat_w_ros[0].detach().cpu().numpy())
    K = cam.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float64)
    X, Y, Z = R.T @ (np.asarray(p_w, dtype=np.float64) - t)
    if Z <= 1e-6:
        return None
    return K[0, 0] * X / Z + K[0, 2], K[1, 1] * Y / Z + K[1, 2]


def draw_cross(d, xy, origin, up, dims, color, r=9):
    if xy is None:
        return
    x, y = xy
    w, h = dims
    px, py = origin[0] + x * up, origin[1] + y * up
    if 0 <= x < w and 0 <= y < h:
        d.ellipse([px - r, py - r, px + r, py + r], outline=color, width=2)
        d.line([px - r - 6, py, px - r + 2, py], fill=color, width=2)
        d.line([px + r - 2, py, px + r + 6, py], fill=color, width=2)
        d.line([px, py - r - 6, px, py - r + 2], fill=color, width=2)
        d.line([px, py + r - 2, px, py + r + 6], fill=color, width=2)
    else:
        px = min(max(px, origin[0] + 8), origin[0] + w * up - 8)
        py = min(max(py, origin[1] + 8), origin[1] + h * up - 8)
        d.ellipse([px - 7, py - 7, px + 7, py + 7], outline=color, width=3)
        d.text((px + 10, py - 6), "OUT", fill=color)


def load_font(size):
    for q in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(q, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def to_uint8(t):
    a = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.001 else a, 0, 255).astype(np.uint8)
    return a


def to_rgb_uint8(t):
    value = to_uint8(t)
    if value.ndim != 3 or value.shape[-1] < 3:
        raise ValueError(f"expected a channel-last RGB camera frame, got {value.shape}")
    return value[..., :3]


def main():
    from evaluation.agent_config import load_agent_config
    agent_cfg_path = args.agent_cfg or CFG_YAML
    agent_cfg = load_agent_config(agent_cfg_path)
    configured_algo = agent_cfg["params"].get("algo", {}).get("name", "")
    if configured_algo in {"a2c_continuous_cats", "a2c_continuous_caps", "a2c_continuous_finite"}:
        # CATS, CAPS, and the finite-gradient guard change training only; evaluation
        # uses the stock PPO player and the same network/checkpoint layout.
        agent_cfg["params"]["algo"]["name"] = "a2c_continuous"
    agent_cfg["params"]["seed"] = args.seed
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = 1
    agent_cfg["params"]["config"]["device"] = "cuda:0"
    agent_cfg["params"]["config"]["device_name"] = "cuda:0"

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    wrapped = RlGamesVecEnvWrapper(env, "cuda:0", clip_obs, clip_act)
    vecenv.register("IsaacRlgWrapper",
                    lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()
    print(">>> 체크포인트 로드: %s" % args.checkpoint, flush=True)
    print(">>> 에이전트 설정: %s" % agent_cfg_path, flush=True)
    print(">>> 평가 알고리즘: %s" % agent_cfg["params"]["algo"]["name"], flush=True)

    robot = raw.scene["robot"]
    full_cam = raw.scene["full_camera"]
    pol_cam = raw.scene["tiled_camera"]
    hand_idx = robot.body_names.index("panda_hand")
    arm_joint_ids, _ = robot.find_joints("panda_joint.*")
    control_dt = float(getattr(raw, "step_dt", 1.0 / args.fps))

    fh, fw = full_cam.image_shape[0], full_cam.image_shape[1]
    uses_standard_layout = args.video_layout == "policy_input_high_res"
    include_action_trace = uses_standard_layout or args.control_trace
    if uses_standard_layout:
        if (fw, fh) != (FULL_WIDTH, FULL_HEIGHT):
            raise ValueError(f"policy_input_high_res requires a {FULL_WIDTH}x{FULL_HEIGHT} full camera")
        W, H = STANDARD_VIDEO_WIDTH, INPUT_ROW_HEIGHT
        pol_side = None
        pol_org = None
        full_org = None
    elif args.video_layout == "high_res_dual":
        if (fw, fh) != (FULL_WIDTH, FULL_HEIGHT):
            raise ValueError(f"high_res_dual requires a {FULL_WIDTH}x{FULL_HEIGHT} full camera")
        W, H = DUAL_WIDTH, FULL_HEIGHT
        pol_side = POLICY_VIEW_SIDE
        pol_org = (0, 0)
        full_org = (POLICY_VIEW_SIDE + GUTTER, 0)
    else:
        pol_side = 84 * POLICY_UP
        PANEL = pol_side + 36
        W, H = fw + PANEL, max(fh, PANEL)
        W += W % 2
        H += H % 2
        full_org = (0, (H - fh) // 2)
        pol_org = (fw + (PANEL - pol_side) // 2, (H - pol_side) // 2)

    video_only_path = f"{args.out}.video_only.tmp.mp4" if include_action_trace else args.out
    writer = imageio_ffmpeg.write_frames(video_only_path, (W, H), fps=args.fps,
                                         macro_block_size=1, quality=8, ffmpeg_log_level="error")
    writer.send(None)
    f_big, f_small = load_font(26), load_font(20)
    GREEN, WHITE, GREY, YELLOW, RED = ((120, 230, 150), (235, 235, 235),
                                       (150, 160, 175), (250, 200, 90), (235, 110, 110))

    obs = wrapped.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)

    per_target = []          # 목표별 최소거리
    per_target_distance_trace = []
    per_target_hold_success = {f"lt_{int(t * 100)}cm": [] for t in HOLD_THRESHOLDS}
    per_target_energy = []
    per_target_power = []
    per_target_smoothness = []
    # 목표추종 기울기용: 지령 목표 위치와, 그 목표에서 손목이 가장 가까웠던 순간의 위치.
    # 정책이 마커를 읽으면 두 점이 함께 움직이고(기울기 b≈1), 마커를 못 읽고 고정
    # 궤적만 그리면 도달점이 목표와 무관해진다(b≈0). base 의 실패 서명을 눈대중이
    # 아니라 회귀계수 하나로 잡는다.
    cmd_xyz, hit_xyz = [], []
    cur_min, cur_hit, prev_cmd, n_t = 1e9, None, None, 0
    cur_torques, cur_velocities = [], []
    cur_actions = []
    cur_distances = []
    all_actions = []
    step = 0
    while n_t <= args.targets:
        cmd = raw.command_manager.get_command("ee_pose")
        if prev_cmd is None or float((cmd[0, :3] - prev_cmd).abs().sum()) > 1e-6:
            if prev_cmd is not None:
                per_target.append(cur_min)
                distance_trace = np.asarray(cur_distances, dtype=np.float64)
                per_target_distance_trace.append(distance_trace.tolist())
                for threshold in HOLD_THRESHOLDS:
                    key = f"lt_{int(threshold * 100)}cm"
                    per_target_hold_success[key].append(
                        hold_success(
                            distance_trace,
                            threshold,
                            HOLD_DURATION_SEC,
                            control_dt,
                        )
                    )
                cmd_xyz.append(prev_cmd.detach().cpu().numpy().copy())
                hit_xyz.append(cur_hit if cur_hit is not None else np.zeros(3))
                energy, mean_power = mechanical_energy(
                    np.asarray(cur_torques), np.asarray(cur_velocities), control_dt
                )
                smoothness = arm_smoothness(
                    np.asarray(cur_actions), fs=1.0 / control_dt
                )
                per_target_energy.append(energy)
                per_target_power.append(mean_power)
                per_target_smoothness.append(smoothness)
                print("    목표 %2d  최소거리 %.4f m  smoothness %.6f  에너지 %.3f J  평균전력 %.3f W  %s" % (
                    n_t, cur_min, smoothness, energy, mean_power,
                    "  ".join("%.0fcm:%s" % (t * 100, "O" if cur_min < t else "X")
                               for t in THRESHOLDS)), flush=True)
                if len(per_target) >= args.targets:
                    break
            n_t += 1
            cur_min, cur_hit = 1e9, None
            cur_torques, cur_velocities = [], []
            cur_actions = []
            cur_distances = []
            prev_cmd = cmd[0, :3].clone()

        policy_frame = latest_policy_rgb(to_uint8(obs))
        full_frame = to_rgb_uint8(full_cam.data.output["rgb"][0])
        with torch.no_grad():
            action = agent.get_action(obs, is_deterministic=True)
        obs, _, dones, _ = wrapped.step(action)
        if isinstance(obs, dict):
            obs = obs["obs"]
        if dones.any():
            agent.reset()

        cur_torques.append(
            robot.data.applied_torque[0, arm_joint_ids].detach().cpu().numpy().copy()
        )
        cur_velocities.append(
            robot.data.joint_vel[0, arm_joint_ids].detach().cpu().numpy().copy()
        )
        cur_actions.append(action.detach().cpu().numpy().reshape(-1).copy())
        all_actions.append(action.detach().cpu().numpy().reshape(-1).copy())

        tgt_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                            cmd[:, :3], cmd[:, 3:7])
        tw = tgt_w[0].detach().cpu().numpy()
        hw = robot.data.body_pos_w[0, hand_idx].detach().cpu().numpy()
        dist = float(np.linalg.norm(tw - hw))
        if dist < cur_min:
            cur_min = dist
            cur_hit = hw.copy()      # 이 목표에서 가장 가까웠던 순간의 손목 위치
        cur_distances.append(dist)

        if uses_standard_layout:
            canvas = Image.fromarray(compose_policy_input_high_res(policy_frame, full_frame))
        elif args.video_layout == "high_res_dual":
            canvas = Image.fromarray(compose_high_res_dual(full_frame))
        else:
            canvas = Image.new("RGB", (W, H), (14, 16, 20))
            canvas.paste(Image.fromarray(full_frame), full_org)
            canvas.paste(Image.fromarray(to_rgb_uint8(pol_cam.data.output["rgb"][0]))
                         .resize((pol_side, pol_side), Image.NEAREST), pol_org)
        d = ImageDraw.Draw(canvas)
        full_target = project_world(full_cam, tw)
        full_hand = project_world(full_cam, hw)
        if uses_standard_layout:
            # Diagnostic only: the policy has already consumed its unmodified image.
            draw_cross(d, full_target, FULL_CAMERA_ORIGIN, 1, (fw, fh), GREEN)
        elif args.video_layout == "high_res_dual":
            d.rectangle((0, 0, W, 42), fill=(10, 12, 16))
            d.text((10, 7), "HIGH-RES POLICY FOV", fill=WHITE, font=f_big)
            d.text((full_org[0] + 10, 7), "FULL CAMERA VIEW", fill=WHITE, font=f_big)
            draw_cross(d, full_to_policy_pixel(full_target), pol_org, 1,
                       (POLICY_VIEW_SIDE, POLICY_VIEW_SIDE), GREEN, r=9)
            draw_cross(d, full_to_policy_pixel(full_hand), pol_org, 1,
                       (POLICY_VIEW_SIDE, POLICY_VIEW_SIDE), YELLOW, r=6)
            draw_cross(d, full_target, full_org, 1, (fw, fh), GREEN)
            draw_cross(d, full_hand, full_org, 1, (fw, fh), YELLOW, r=6)
        else:
            draw_cross(d, full_target, full_org, 1, (fw, fh), GREEN)
            draw_cross(d, project_world(pol_cam, tw), pol_org, POLICY_UP, (84, 84), GREEN, r=14)
            draw_cross(d, full_hand, full_org, 1, (fw, fh), YELLOW, r=6)
            d.text((10, 8), args.label or "BASE policy", fill=WHITE, font=f_big)
            d.text((10, 40), "green = target   yellow = panda_hand (wrist, 5.84cm behind fingertips)",
                   fill=GREY, font=f_small)
        ok = cur_min < 0.05
        if uses_standard_layout:
            pass
        elif args.video_layout == "high_res_dual":
            d.rectangle((0, H - 38, W, H), fill=(10, 12, 16))
            d.text((10, H - 34), "target %d/%d  dist %.3f m  best %.3f m  %s" % (
                n_t, args.targets, dist, cur_min, "SUCCESS <5cm" if ok else ""),
                fill=GREEN if ok else WHITE, font=f_big)
        else:
            d.text((10, H - 62), "target %d / %d" % (n_t, args.targets), fill=GREEN, font=f_big)
            d.text((10, H - 34), "dist %.3f m   best %.3f m   %s" % (
                dist, cur_min, "SUCCESS <5cm" if ok else ""),
                fill=GREEN if ok else WHITE, font=f_big)
            d.text((pol_org[0], pol_org[1] - 26), "POLICY VIEW 84x84", fill=WHITE, font=f_small)
        writer.send(np.asarray(canvas).tobytes())
        step += 1

    writer.close()
    action_array = np.asarray(all_actions, dtype=np.float64)
    if action_array.ndim != 2 or action_array.shape[1] != len(REACH_ACTION_LABELS):
        raise RuntimeError(
            f"Reach evaluator expected {len(REACH_ACTION_LABELS)} policy actions, got {action_array.shape}"
        )
    if include_action_trace:
        reader = imageio_ffmpeg.read_frames(video_only_path, pix_fmt="rgb24")
        metadata = next(reader)
        source_width, source_height = metadata["size"]
        if (source_width, source_height) != (W, H):
            raise RuntimeError(
                f"unexpected temporary video size {(source_width, source_height)}, expected {(W, H)}"
            )
        output_writer = imageio_ffmpeg.write_frames(
            args.out,
            (W, H + ACTION_TRACE_HEIGHT),
            fps=args.fps,
            macro_block_size=1,
            quality=8,
            ffmpeg_log_level="error",
        )
        output_writer.send(None)
        frames_written = 0
        try:
            for frame_index, frame_bytes in enumerate(reader):
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(H, W, 3)
                output_writer.send(
                    compose_action_trace(
                        frame,
                        action_array,
                        frame_index,
                        action_labels=REACH_ACTION_LABELS,
                    ).tobytes()
                )
                frames_written += 1
        finally:
            reader.close()
            output_writer.close()
        os.remove(video_only_path)
        if frames_written != len(action_array):
            raise RuntimeError(
                f"video/action length mismatch: {frames_written} frames vs {len(action_array)} actions"
            )
    arr = np.array(per_target)
    print("\n>>> 성공률 (목표 %d개, 판정=목표 유효구간 내 최소거리)" % len(arr), flush=True)
    for t in THRESHOLDS:
        print("      < %4.0f cm : %4.0f%%  (%d/%d)" % (
            t * 100, 100.0 * (arr < t).mean(), int((arr < t).sum()), len(arr)), flush=True)
    print("      최소거리 중앙값 %.4f m / 평균 %.4f m / 최악 %.4f m"
          % (float(np.median(arr)), float(arr.mean()), float(arr.max())), flush=True)
    smoothness_values = np.asarray(per_target_smoothness, dtype=np.float64)
    approx_arm_smoothness = float(smoothness_values.mean())
    print("      arm smoothness %.6f +/- %.6f"
          % (approx_arm_smoothness, float(smoothness_values.std(ddof=0))), flush=True)
    energy_values = np.asarray(per_target_energy, dtype=np.float64)
    power_values = np.asarray(per_target_power, dtype=np.float64)
    approx_arm_energy_j = float(energy_values.mean())
    approx_arm_mean_power_w = float(power_values.mean())
    print(
        "      approximate arm energy %.3f +/- %.3f J / mean power %.3f +/- %.3f W"
        % (
            approx_arm_energy_j,
            float(energy_values.std(ddof=0)),
            approx_arm_mean_power_w,
            float(power_values.std(ddof=0)),
        ),
        flush=True,
    )

    # ── 목표추종 기울기 ────────────────────────────────────────────────
    # 축마다 hit = a + b·cmd 를 최소제곱으로 맞춘다. b 를 세 축 평균해 보고한다.
    # b≈1  목표를 그대로 따라간다 (마커를 읽고 있다)
    # b≈0  목표가 어디든 같은 곳으로 간다 (CNN 출력이 상수로 붕괴)
    slope_summary = None
    if len(cmd_xyz) >= 3:
        C = np.asarray(cmd_xyz, dtype=np.float64)
        Hh = np.asarray(hit_xyz, dtype=np.float64)
        bs, r2s = [], []
        for k in range(3):
            x, y = C[:, k], Hh[:, k]
            vx = x.var()
            if vx < 1e-9:
                continue
            b = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
            pred = y.mean() + b * (x - x.mean())
            ss = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 1e-12 else float("nan")
            bs.append(b)
            r2s.append(r2)
        if bs:
            print("      목표추종 기울기 b = %.3f  (축별 %s, R2 평균 %.3f)"
                  % (float(np.mean(bs)), " ".join("%.2f" % v for v in bs),
                     float(np.nanmean(r2s))), flush=True)
            slope_summary = {
                "mean": float(np.mean(bs)),
                "per_axis": [float(v) for v in bs],
                "r2_mean": float(np.nanmean(r2s)),
            }

    if args.summary:
        summary = {
            "task": args.task,
            "checkpoint": args.checkpoint,
            "agent_cfg": agent_cfg_path,
            "seed": int(args.seed),
            "num_targets": int(len(arr)),
            "metric_definition": {
                "performance": "minimum panda_hand-to-target distance per target window",
                "success_rate": "fraction of target windows continuously below 2, 5, or 10 cm for at least 1.0 s",
                "minimum_distance_success_rate": "fraction of target windows whose minimum distance is below each threshold",
                "arm_smoothness": "coordinate-wise frequency-weighted score averaged over seven arm actions",
                "approx_arm_energy_j": "dt * sum_t sum_j abs(applied_torque[t,j] * joint_velocity[t,j])",
            },
            "min_distance_m": {
                "per_target": [float(v) for v in arr],
                "median": float(np.median(arr)),
                "mean": float(arr.mean()),
                "max": float(arr.max()),
            },
            "minimum_distance_success_rate": {
                f"lt_{int(t * 100)}cm": float((arr < t).mean())
                for t in THRESHOLDS
            },
            "distance_trace_m": {
                "control_dt": control_dt,
                "per_target": per_target_distance_trace,
            },
            "hold_success_1s": {
                key: {
                    "required_duration_s": HOLD_DURATION_SEC,
                    "per_target": values,
                    "rate": float(np.mean(values)),
                }
                for key, values in per_target_hold_success.items()
            },
            "success_rate": {
                key: float(np.mean(values))
                for key, values in per_target_hold_success.items()
            },
            "approx_arm_energy_j": {
                "per_target": [float(v) for v in energy_values],
                "mean": approx_arm_energy_j,
                "std": float(energy_values.std(ddof=0)),
            },
            "arm_smoothness": {
                "per_target": [float(v) for v in smoothness_values],
                "mean": approx_arm_smoothness,
                "std": float(smoothness_values.std(ddof=0)),
            },
            "approx_arm_mean_power_w": {
                "per_target": [float(v) for v in power_values],
                "mean": approx_arm_mean_power_w,
                "std": float(power_values.std(ddof=0)),
            },
            "target_following": slope_summary,
            "policy_actions": {
                "definition": "deterministic seven-dimensional policy output before JointPositionActionCfg scale/offset",
                "control_dt": control_dt,
                "values": action_array.tolist(),
            },
        }
        summary_dir = os.path.dirname(os.path.abspath(args.summary))
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(">>> SUMMARY WRITTEN %s" % args.summary, flush=True)
    print(">>> EVAL DONE %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6), flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
