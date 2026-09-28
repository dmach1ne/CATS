"""Ten physical environments, fixed paired goals, independent pre-reset metrics."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import eval_pregrasped_placement as old


def collect(args, raw, wrapped, player, schedule):
    import torch
    from isaaclab.utils.math import subtract_frame_transforms
    from er_isaac.tasks.manager_based.er_isaac.mdp.placement import get_eval_snapshot
    from high_res_video import latest_policy_rgb
    from placement_first10_video import save_first10_video

    if raw.num_envs != 10 or raw.max_episode_length != 90 or abs(raw.step_dt - 1/30) > 1e-9:
        raise ValueError('Parallel evaluation requires ten environments, 90 steps, 30Hz')
    trials, archives = [], {}
    output = Path(args.out)
    captured = {}
    compute = raw.reward_manager.compute

    def capture(dt):
        reward = compute(dt)
        snapshot = get_eval_snapshot(raw)
        robot, obj = raw.scene['robot'], raw.scene['object']
        pos, quat = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w,
                                             obj.data.root_pos_w, obj.data.root_quat_w)
        captured['frames'] = []
        for i in range(10):
            frame = old.snapshot_frame({k: v[i:i+1] for k, v in snapshot.items()})
            frame['cube_bottom_height_above_table_m'] = old.cube_bottom_above_table(
                pos[i, 2].item(), quat[i].detach().cpu().numpy())
            frame['cube_center_height_above_table_m'] = pos[i, 2].item() - .026
            captured['frames'].append(frame)
        for key, value in [('rewards', reward), ('done', raw.reset_buf),
                           ('terminated', raw.reset_terminated), ('truncated', raw.reset_time_outs)]:
            captured[key] = value.detach().cpu().numpy().copy().reshape(-1)
        return reward

    raw.reward_manager.compute = capture
    try:
        for seed in args.eval_seeds:
            archive = {}
            archive_path = output.with_name(f'{output.stem}.seed{seed}.traces.npz')
            entries = [r for r in schedule['trials'] if r['seed'] == seed]
            if len(entries) != args.trials:
                raise ValueError('Fixed goal schedule has incorrect trial count')
            for start in range(0, args.trials, 10):
                batch = entries[start:start+10]
                raw._eval_goal_batch = torch.tensor([r['goal_position_b'] for r in batch],
                                                    device=raw.device, dtype=torch.float32)
                observation = wrapped.reset()
                observation = observation['obs'] if isinstance(observation, dict) else observation
                player.reset()
                player.get_batch_size(observation, 10)
                active = np.ones(10, dtype=bool)
                holds = np.zeros((10, 3))
                frames = [[] for _ in range(10)]
                actions = [[] for _ in range(10)]
                observations = [[] for _ in range(10)]
                targets = [[] for _ in range(10)]
                movies = [[] for _ in range(10)]
                returns = np.zeros(10)
                terminal = {}
                for step in range(90):
                    captured.clear()
                    with torch.no_grad():
                        policy_observation = observation.detach().cpu().numpy().copy()
                        if policy_observation.shape != (10, 84 * 84 * 3 + 26):
                            raise ValueError(
                                f'Expected ten RGB84+raw26 observations, got {policy_observation.shape}'
                            )
                        target_xyz = raw.command_manager.get_command('object_pose')[:, :3].detach().cpu().numpy().copy()
                        if target_xyz.shape != (10, 3) or not np.isfinite(target_xyz).all():
                            raise ValueError(f'Expected finite base-frame targets (10,3), got {target_xyz.shape}')
                        if not np.allclose(policy_observation[:, -3:], target_xyz, atol=1e-7, rtol=0):
                            raise RuntimeError('Observation goal XYZ and object_pose command disagree')
                        action = player.get_action(observation, is_deterministic=True)
                        values = action.detach().cpu().numpy().copy()
                        if values.shape != (10, 8) or not np.isfinite(values).all():
                            raise ValueError('Parallel policy must return finite actions (10,8)')
                        if seed == args.eval_seeds[0] and start == 0:
                            pixels = observation.detach().cpu().numpy()
                            full = raw.scene['full_camera'].data.output['rgb'][..., :3].detach().cpu().numpy()
                            for i in np.flatnonzero(active):
                                camera = full[i]
                                if camera.dtype != np.uint8:
                                    camera = np.clip(camera * (255 if camera.max() <= 1.001 else 1), 0, 255).astype(np.uint8)
                                movies[i].append((latest_policy_rgb(pixels[i]).copy(), camera.copy()))
                        observation, _, done, _ = wrapped.step(action)
                        observation = observation['obs'] if isinstance(observation, dict) else observation
                    if 'frames' not in captured:
                        raise RuntimeError('Missing parallel pre-reset capture')
                    if not np.array_equal(done.detach().cpu().numpy().reshape(-1).astype(bool), captured['done'].astype(bool)):
                        raise RuntimeError('Wrapper done and pre-reset capture disagree')
                    for i in np.flatnonzero(active):
                        frame = captured['frames'][i]
                        if not np.allclose(frame['goal_position_b'], batch[i]['goal_position_b'], atol=1e-7, rtol=0):
                            raise RuntimeError('Goal changed within an active trial')
                        holds[i] = old.update_old_success(frame, holds[i], raw.step_dt)
                        frames[i].append(frame)
                        actions[i].append(values[i])
                        if seed == args.eval_seeds[0] and start == 0:
                            observations[i].append(policy_observation[i])
                            targets[i].append(target_xyz[i])
                        returns[i] += captured['rewards'][i]
                        if captured['done'][i]:
                            active[i] = False
                            terminal[i] = (bool(captured['terminated'][i]), bool(captured['truncated'][i]))
                    if not active.any():
                        break
                if active.any():
                    raise RuntimeError('Some parallel episodes did not terminate within 90 steps')
                for i, entry in enumerate(batch):
                    trial = entry['trial']
                    torques = np.asarray([f['arm_applied_torque'] for f in frames[i]])
                    velocities = np.asarray([f['arm_joint_velocity'] for f in frames[i]])
                    energy, _ = old.mechanical_energy(torques, velocities, raw.step_dt)
                    row = dict(seed=seed, trial=trial, episode_return=float(returns[i]),
                        duration_s=len(frames[i])*raw.step_dt,
                        min_xy_error_m=min(f['xy_error_m'] for f in frames[i]),
                        final_xy_error_m=frames[i][-1]['xy_error_m'],
                        arm_smoothness=old.arm_smoothness(np.asarray(actions[i]), fs=30),
                        approx_arm_energy_j=energy, terminated=terminal[i][0], truncated=terminal[i][1],
                        trace=frames[i], environment_index=i)
                    for key in ('success_2cm_1s', 'success_5cm_1s', 'success_10cm_1s'):
                        row[key] = any(f[key] for f in frames[i])
                    for key in ('cube_bottom_height_above_table_m', 'cube_center_height_above_table_m', 'cube_rise_m'):
                        row['max_' + key] = max(f[key] for f in frames[i])
                    for key, value in [('actions', actions[i]), ('torques', torques), ('velocities', velocities)]:
                        archive[f'trial_{trial}_{key}'] = np.asarray(value)
                    trials.append(row)
                old.save_archive(archive_path, archive)
                archives[str(seed)] = str(archive_path)
                old.save_report(args, trials, args.eval_seeds, 'partial', archives=archives)
                print(f'placement_parallel_eval={len(trials)}/{len(schedule["trials"])}', flush=True)
                if seed == args.eval_seeds[0] and start == 0:
                    joined, values, boundaries = [], [], []
                    images, proprio, target_values, trial_ids, step_ids = [], [], [], [], []
                    for i in range(10):
                        joined.extend(movies[i])
                        values.extend(actions[i])
                        boundaries.append(len(values))
                        trial_observations = np.asarray(observations[i], dtype=np.float32)
                        trial_targets = np.asarray(targets[i], dtype=np.float32)
                        images.append(trial_observations[:, :84 * 84 * 3].reshape(-1, 84, 84, 3))
                        proprio.append(trial_observations[:, -26:])
                        target_values.append(trial_targets)
                        trial_ids.extend([i + 1] * len(trial_observations))
                        step_ids.extend(range(len(trial_observations)))
                    save_first10_video(output.with_name('first10'), joined, values, boundaries)
                    observation_path = output.with_name('first10.observations.npz')
                    old.save_archive(observation_path, {
                        'image': np.concatenate(images),
                        'proprio': np.concatenate(proprio),
                        'target_xyz': np.concatenate(target_values),
                        'action': np.asarray(values, dtype=np.float32),
                        'trial_boundaries': np.asarray(boundaries, dtype=np.int64),
                        'trial_id': np.asarray(trial_ids, dtype=np.int64),
                        'step_in_trial': np.asarray(step_ids, dtype=np.int64),
                        'policy_rate_hz': np.asarray(30., dtype=np.float32),
                    })
                    print(f'placement_first10_observations={observation_path}', flush=True)
                    print('placement_first10_video=complete', flush=True)
        old.save_report(args, trials, args.eval_seeds, 'complete', archives=archives)
    except BaseException:
        old.save_report(args, trials, args.eval_seeds, 'failed', traceback.format_exc(), archives)
        raise
    finally:
        raw.reward_manager.compute = compute


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'agent-cfg', 'task', 'out', 'goal-schedule'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=list(old.DEFAULT_EVAL_SEEDS))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.trials != 50 or args.eval_seeds != list(old.DEFAULT_EVAL_SEEDS):
        parser.error('Use the frozen five-seed, fifty-trial recipe')
    args.device = args.device or 'cuda:0'
    args.headless = True
    schedule = json.loads(Path(args.goal_schedule).read_text())
    old.PROTOCOL.update(parallel_environments=10, goal_assignment='paired_frozen_BASE424225_goals',
                        initial_grasp='per_environment_physical_validation_then_snapshot_restore')
    app = env = None
    try:
        cfg = old.prepare_agent(args.agent_cfg, args.checkpoint, args.eval_seeds[0], args.device)
        cfg['params']['config']['num_actors'] = 10
        app = AppLauncher(args).app
        import gymnasium as gym
        import torch
        from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        import er_isaac.tasks
        from er_isaac.tasks.manager_based.er_isaac.mdp.placement import PlacementGoalCommand

        def sample_fixed(command, env_ids):
            goals = getattr(command._env, '_eval_goal_batch', None)
            if goals is None:
                raise RuntimeError('Reset attempted before fixed goal assignment')
            command.pose_command_b[env_ids, :3] = goals[env_ids]
            command.pose_command_b[env_ids, 3:] = goals.new_tensor([1., 0., 0., 0.])

        PlacementGoalCommand._resample_command = sample_fixed
        sys.path.insert(0, str(old.EVAL_DIR.parent / 'rl_games'))
        from models.visual_proprio_network import register_visual_proprio_networks
        register_visual_proprio_networks()
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=10)
        env_cfg.seed = args.eval_seeds[0]
        if tuple(env_cfg.observations.policy.image_proprio.params.get('image_shape', (84,84,3))) != (84,84,3):
            raise ValueError('Parallel evaluator requires the saved RGB84 observation')
        env = gym.make(args.task, cfg=env_cfg)
        wrapped = RlGamesVecEnvWrapper(env, args.device, cfg['params']['env'].get('clip_observations', float('inf')),
                                       cfg['params']['env'].get('clip_actions', float('inf')))
        vecenv.register('IsaacRlgWrapper', lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register('rlgpu', {'vecenv_type':'IsaacRlgWrapper', 'env_creator':lambda **kw: wrapped})
        runner = Runner()
        runner.load(cfg)
        player = runner.create_player()
        player.restore(args.checkpoint)
        collect(args, env.unwrapped, wrapped, player, schedule)
    except BaseException:
        output = Path(args.out)
        if not output.exists():
            old.save_report(args, [], args.eval_seeds, 'failed', traceback.format_exc())
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
