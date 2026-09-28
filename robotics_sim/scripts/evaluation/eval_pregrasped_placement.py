"""Deterministic placement evaluation with pre-reset capture and incremental persistence."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
from agent_config import load_agent_config
from lift_metrics import arm_smoothness, mechanical_energy
from placement_legacy_eval_metrics import update_old_success, cube_bottom_above_table

DEFAULT_EVAL_SEEDS = (652165, 709797, 242820, 579787, 70701)
PROTOCOL = dict(name='pregrasped_placement_front25cm_release_hold1s_v1',
                xy_thresholds_m=[.02, .05, .10], hold_s=1.0, rest_height_m=.0505,
                reward_protocol='placesphere_front25_transport10_lower_near5_v1',
                rest_height_tolerance_m=.005, open_width_m=.06,
                table_contact_required=True, finger_contacts_absent=True,
                success_requires_lift=False, success_rule="old_rest_release_hold_no_lift_gate", success_aggregation='episode_ever',
                orientation_required=False, goal_radius_m=.25, goal_sampling='uniform_front_semicircumference_x_positive',
                goal_angle_range_rad=[-math.pi/2, math.pi/2], movement_lift_ramp_m=.10, near_goal_xy_m=.05,
                episode_length_s=3., policy_rate_hz=30., preparation_included=False,
                action_trace='raw_policy_output_before_wrapper_clipping',
                energy_sampling='pre_reset_policy_rate_approximation')


PROTOCOL.update(height_reference="table top in robot base", table_top_base_z_m=.026,
                cube_side_m=.049, cube_bottom="orientation-aware lowest corner",
                success_sampling_hz=30., video_eval_seed=652165, video_trials=10,
                video_layout="latest policy input / high-res / arm7+gripper1 trace")

def summarize(trials):
    result = {'num_trials': len(trials)}
    for key in ('success_2cm_1s', 'success_5cm_1s', 'success_10cm_1s'):
        result[key] = float(np.mean([row[key] for row in trials])) if trials else None
    for key in ('max_cube_bottom_height_above_table_m', 'max_cube_center_height_above_table_m', 'max_cube_rise_m', 'episode_return', 'duration_s', 'min_xy_error_m',
                'final_xy_error_m', 'arm_smoothness', 'approx_arm_energy_j'):
        values = np.asarray([row[key] for row in trials], dtype=float)
        result[key] = {'mean': float(values.mean()), 'std': float(values.std())} if trials else None
    return result


def build_report(args, trials, seeds, status, error=None, archives=None):
    return dict(protocol=PROTOCOL, task=args.task,
                checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
                agent_cfg=str(Path(args.agent_cfg).expanduser().resolve()),
                evaluation_seeds=list(seeds), trials_per_seed=args.trials,
                expected_num_trials=len(seeds) * args.trials, trials=trials,
                summary=summarize(trials), status=status, error=error,
                per_seed_summaries=[dict(seed=seed, summary=summarize(
                    [row for row in trials if row['seed'] == seed])) for seed in seeds],
                raw_trace_archives=archives or {})


def save_report(args, trials, seeds, status, error=None, archives=None):
    path = Path(args.out).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_report(args, trials, seeds, status, error, archives),
                         indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(payload, encoding='utf-8')
    temporary.replace(path)


def save_archive(path, archive):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **archive)
    temporary.replace(path)


def prepare_agent(path, checkpoint, seed, device):
    cfg = load_agent_config(path)
    network = cfg['params']['network']
    if (network.get('fusion') != 'compact_proprio_goal_fixed_scale'
            or network.get('proprio_size') != 26 or network.get('proprio_fixed_scale') is not True
            or tuple(network.get('image_shape', (84, 84, 3))) != (84, 84, 3)):
        raise ValueError('Placement requires saved RGB84/GoalFixedScale/raw26 agent configuration')
    cfg['params']['seed'] = seed
    cfg['params']['algo']['name'] = 'a2c_continuous'
    cfg['params']['load_checkpoint'] = True
    cfg['params']['load_path'] = str(Path(checkpoint).expanduser().resolve())
    cfg['params']['config'].update(num_actors=1, device=device, device_name=device, multi_gpu=False)
    return cfg


def snapshot_frame(snapshot):
    required = {'xy_error_m', 'success_2cm_1s', 'success_5cm_1s', 'success_10cm_1s'}
    if not required.issubset(snapshot):
        raise ValueError(f'Placement snapshot missing {sorted(required - snapshot.keys())}')
    result = {}
    for key, value in snapshot.items():
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 0 or value.shape[0] != 1:
            raise ValueError(f'Placement snapshot {key} must have one environment: {value.shape}')
        if not np.isfinite(value).all():
            raise ValueError(f'Placement snapshot {key} must be finite')
        result[key] = value.reshape(-1)[0].item() if value.size == 1 else value[0].tolist()
    for key in ('success_2cm_1s', 'success_5cm_1s', 'success_10cm_1s'):
        if not isinstance(result[key], bool):
            raise ValueError(f'Placement snapshot {key} must be boolean')
    return result


def validate_child(report, args, seed, complete=True):
    expected = build_report(args, [], [seed], 'complete')
    for key in ('task', 'checkpoint', 'agent_cfg', 'evaluation_seeds', 'protocol', 'trials_per_seed'):
        if report.get(key) != expected[key]:
            raise ValueError(f'Placement child {seed}: mismatched {key}')
    if complete and report.get('status') != 'complete':
        raise ValueError(f'Placement child {seed}: not complete')
    rows = report.get('trials', [])
    if len(rows) > args.trials or (complete and len(rows) != args.trials):
        raise ValueError(f'Placement child {seed}: incorrect trial count')
    if [row.get('trial') for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f'Placement child {seed}: invalid trial IDs')
    if any(row.get('seed') != seed for row in rows):
        raise ValueError(f'Placement child {seed}: trial seed mismatch')


def run_sweep(args):
    trials, archives = [], {}
    seeds = list(args.eval_seeds)
    path = Path(args.out).expanduser().resolve()
    save_report(args, trials, seeds, 'running')
    try:
        for seed in seeds:
            child = path.with_name(f'{path.stem}.seed{seed}.json')
            # Never accept stale output if this invocation fails before startup.
            child.unlink(missing_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), '--checkpoint', args.checkpoint,
                       '--agent-cfg', args.agent_cfg, '--task', args.task, '--trials', str(args.trials),
                       '--seed', str(seed), '--out', str(child), '--device', args.device,
                       '--headless', '--enable_cameras']
            try:
                subprocess.run(command, check=True)
            except BaseException:
                if child.exists():
                    report = json.loads(child.read_text())
                    validate_child(report, args, seed, complete=False)
                    trials.extend(report['trials'])
                    archives.update(report.get('raw_trace_archives', {}))
                raise
            report = json.loads(child.read_text())
            validate_child(report, args, seed)
            trials.extend(report['trials'])
            archives.update(report.get('raw_trace_archives', {}))
            save_report(args, trials, seeds, 'partial', archives=archives)
        save_report(args, trials, seeds, 'complete', archives=archives)
    except BaseException:
        save_report(args, trials, seeds, 'failed', traceback.format_exc(), archives)
        raise


def collect_trials(args, raw, wrapped, player, get_snapshot, trials, archive):
    import torch
    if raw.num_envs != 1 or not math.isclose(float(raw.step_dt), 1 / 30) or raw.max_episode_length != 90:
        raise ValueError('Placement evaluation requires one environment, 30 Hz and 90 policy steps')
    dt = float(raw.step_dt)
    archive_path = Path(args.out).expanduser().resolve().with_suffix('.traces.npz')
    archives = {str(args.seed): str(archive_path)}
    captured = {}
    old_hold = np.zeros(3)
    video_frames, video_actions, video_boundaries = [], [], []
    compute = raw.reward_manager.compute

    def capture_compute(dt):
        reward = compute(dt=dt)
        frame = snapshot_frame(get_snapshot(raw))
        from isaaclab.utils.math import subtract_frame_transforms
        robot, obj = raw.scene['robot'], raw.scene['object']
        pos, quat = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                             obj.data.root_pos_w, obj.data.root_quat_w)
        frame['cube_bottom_height_above_table_m'] = cube_bottom_above_table(
            float(pos[0,2].item()), quat[0].detach().cpu().numpy())
        frame['cube_center_height_above_table_m'] = float(pos[0,2].item()) - .026
        for key in ('arm_applied_torque', 'arm_joint_velocity'):
            if np.asarray(frame.get(key)).shape != (7,):
                raise ValueError(f'Placement snapshot requires pre-reset {key} shape (1,7)')
        captured.update(frame=frame, reward=float(reward.detach().cpu().reshape(-1).item()),
                        done=bool(raw.reset_buf[0].item()),
                        terminated=bool(raw.reset_terminated[0].item()),
                        truncated=bool(raw.reset_time_outs[0].item()))
        return reward

    raw.reward_manager.compute = capture_compute
    try:
        observation = wrapped.reset()
        observation = observation['obs'] if isinstance(observation, dict) else observation
        for trial in range(args.trials):
            old_hold[:] = 0
            player.reset()
            player.get_batch_size(observation, 1)
            actions, torques, velocities, frames = [], [], [], []
            total = 0.
            transition_error = None
            for _ in range(raw.max_episode_length):
                captured.clear()
                with torch.no_grad():
                    action = player.get_action(observation, is_deterministic=True)
                    action_np = action.detach().cpu().numpy().reshape(-1).copy()
                    if action_np.shape != (8,) or not np.isfinite(action_np).all():
                        raise ValueError('Placement policy must return finite arm7+gripper1 actions')
                    if args.seed == DEFAULT_EVAL_SEEDS[0] and trial < 10:
                        from high_res_video import latest_policy_rgb
                        policy = latest_policy_rgb(observation.detach().cpu().numpy()).copy()
                        full = raw.scene['full_camera'].data.output['rgb'][0,:,:,:3].detach().cpu().numpy()
                        if full.dtype != np.uint8:
                            full = np.clip(full * (255 if full.max() <= 1.001 else 1), 0, 255).astype(np.uint8)
                        video_frames.append((policy, full.copy()))
                        video_actions.append(action_np)
                    try:
                        observation, _, done, _ = wrapped.step(action)
                    except BaseException as error:
                        # Preserve a valid terminal sample if the ensuing auto-reset fails.
                        if not captured.get('done', False):
                            raise
                        transition_error = error
                if 'frame' not in captured:
                    raise RuntimeError('Placement step produced no pre-reset reward snapshot')
                observation = observation['obs'] if isinstance(observation, dict) else observation
                old_hold = update_old_success(captured['frame'], old_hold, dt)
                frames.append(captured['frame'])
                actions.append(action_np)
                torques.append(frames[-1]['arm_applied_torque'])
                velocities.append(frames[-1]['arm_joint_velocity'])
                total += captured['reward']
                if captured['done']:
                    break
                if bool(done.any()):
                    raise RuntimeError('Wrapper done disagrees with pre-reset termination capture')
            else:
                raise RuntimeError('Placement episode did not terminate within 90 policy steps')
            energy, _ = mechanical_energy(np.asarray(torques), np.asarray(velocities), dt)
            if args.seed == DEFAULT_EVAL_SEEDS[0] and trial < 10:
                video_boundaries.append(len(video_actions))
                if trial == 9:
                    from placement_first10_video import save_first10_video
                    save_first10_video(Path(args.out).with_name('first10'), video_frames, video_actions, video_boundaries)
                    video_frames.clear()
            row = dict(seed=args.seed, trial=trial + 1, episode_return=total,
                       max_cube_bottom_height_above_table_m=max(f['cube_bottom_height_above_table_m'] for f in frames),
                       max_cube_center_height_above_table_m=max(f['cube_center_height_above_table_m'] for f in frames),
                       max_cube_rise_m=max(f['cube_rise_m'] for f in frames),
                       duration_s=len(frames) * dt, min_xy_error_m=min(f['xy_error_m'] for f in frames),
                       final_xy_error_m=frames[-1]['xy_error_m'],
                       success_2cm_1s=any(f['success_2cm_1s'] for f in frames),
                       success_5cm_1s=any(f['success_5cm_1s'] for f in frames),
                       success_10cm_1s=any(f['success_10cm_1s'] for f in frames),
                       arm_smoothness=arm_smoothness(np.asarray(actions), fs=1 / dt),
                       approx_arm_energy_j=energy, terminated=captured['terminated'],
                       truncated=captured['truncated'], trace=frames)
            for key, values in (('actions', actions), ('torques', torques), ('velocities', velocities)):
                archive[f'trial_{trial + 1}_{key}'] = np.asarray(values)
            save_archive(archive_path, archive)
            trials.append(row)
            save_report(args, trials, [args.seed], 'partial', archives=archives)
            print(f'placement_eval={trial + 1}/{args.trials}', flush=True)
            if transition_error is not None:
                raise transition_error
    finally:
        raw.reward_manager.compute = compute


def run_single(args, app_launcher):
    trials, archive = [], {}
    app = env = None
    path = Path(args.out).expanduser().resolve()
    archive_path = path.with_suffix('.traces.npz')
    archives = {}
    save_report(args, trials, [args.seed], 'running')
    path.with_suffix('.error.txt').unlink(missing_ok=True)
    try:
        cfg = prepare_agent(args.agent_cfg, args.checkpoint, args.seed, args.device)
        if not Path(args.checkpoint).is_file():
            raise FileNotFoundError(args.checkpoint)
        app = app_launcher(args).app
        import gymnasium as gym
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        import er_isaac.tasks  # noqa: F401
        from er_isaac.tasks.manager_based.er_isaac.mdp.placement import get_eval_snapshot

        sys.path.insert(0, str(EVAL_DIR.parent / 'rl_games'))
        from models.visual_proprio_network import register_visual_proprio_networks
        register_visual_proprio_networks()
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        if tuple(env_cfg.observations.policy.image_proprio.params.get('image_shape', (84, 84, 3))) != (84, 84, 3):
            raise ValueError('Placement task and saved RGB84 network disagree')
        env_cfg.seed = args.seed
        env = gym.make(args.task, cfg=env_cfg)
        wrapped = RlGamesVecEnvWrapper(env, args.device,
                                       cfg['params']['env'].get('clip_observations', math.inf),
                                       cfg['params']['env'].get('clip_actions', math.inf))
        vecenv.register('IsaacRlgWrapper', lambda config_name, num_actors, **kw:
                        RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register('rlgpu', {'vecenv_type': 'IsaacRlgWrapper',
                                            'env_creator': lambda **kw: wrapped})
        runner = Runner()
        runner.load(cfg)
        player = runner.create_player()
        player.restore(args.checkpoint)
        collect_trials(args, env.unwrapped, wrapped, player, get_eval_snapshot, trials, archive)
        archives[str(args.seed)] = str(archive_path)
        # SimulationApp.close can exit the process instead of returning.
        save_report(args, trials, [args.seed], 'complete', archives=archives)
    except BaseException:
        error = traceback.format_exc()
        if archive:
            archives[str(args.seed)] = str(archive_path)
        save_report(args, trials, [args.seed], 'failed', error, archives)
        path.with_suffix('.error.txt').write_text(error, encoding='utf-8')
        raise
    finally:
        already_failing = sys.exc_info()[0] is not None
        try:
            try:
                if env is not None:
                    env.close()
            finally:
                if app is not None:
                    app.close()
        except BaseException:
            error = traceback.format_exc()
            if already_failing:
                print('Placement cleanup also failed:\n' + error, file=sys.stderr, flush=True)
            else:
                save_report(args, trials, [args.seed], 'failed', error, archives)
                path.with_suffix('.error.txt').write_text(error, encoding='utf-8')
                raise
    save_report(args, trials, [args.seed], 'complete', archives=archives)


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--agent-cfg', required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--trials', type=int, default=50)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seed', type=int)
    group.add_argument('--eval-seeds', type=int, nargs='+')
    parser.add_argument('--out', required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    if args.seed is None and args.eval_seeds is None:
        args.eval_seeds = list(DEFAULT_EVAL_SEEDS)
    if args.eval_seeds is not None and len(set(args.eval_seeds)) != len(args.eval_seeds):
        parser.error('--eval-seeds must be unique')
    args.device = args.device or 'cuda:0'
    args.checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    args.agent_cfg = str(Path(args.agent_cfg).expanduser().resolve())
    args.headless = True
    args.enable_cameras = True
    if args.eval_seeds is not None:
        run_sweep(args)
    else:
        run_single(args, AppLauncher)


if __name__ == '__main__':
    main()
