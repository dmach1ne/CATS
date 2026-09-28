#!/usr/bin/env python3
"""Run one PPO method/seed and record end-to-end training cost."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.modules.controller import (
    train_asap,
    train_caps,
    train_cats,
    train_grad,
    train_l2c2,
    train_lipsnetpp,
    train_vanilla,
)
from tests.modules.envs import make_ant_env, make_hopper_env, make_walker_env
from tests.modules.params import alg_args, env_args, env_timestep


METHODS = {
    "base": ("vanilla", train_vanilla),
    "caps": ("caps", train_caps),
    "grad": ("grad", train_grad),
    "l2c2": ("l2c2", train_l2c2),
    "asap": ("asap", train_asap),
    "cats": ("vanilla", train_cats),
    "lpp": ("lipsnetpp", train_lipsnetpp),
}
ENVIRONMENTS = {
    "ant": ("Ant-v5", make_ant_env),
    "hopper": ("Hopper-v5", make_hopper_env),
    "walker": ("Walker2d-v5", make_walker_env),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--environment", choices=("walker",), default="walker")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.environ["CATS_DISABLE_AUXILIARY_METRICS"] = "1"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this timing benchmark")

    config_name, train_function = METHODS[args.method]
    environment_name, make_env = ENVIRONMENTS[args.environment]
    method_root = args.output_root / args.method
    save_dir = method_root / "pths" / args.environment
    log_dir = method_root / "tensorboard_logs" / args.environment
    timing_dir = args.output_root / "timing"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    method_args = alg_args[config_name][args.environment].copy()
    if args.method == "cats":
        method_args["cats_alpha"] = 1.0

    target_steps = env_timestep[args.environment]
    n_envs = env_args[args.environment].get("n_envs", 1)
    n_steps = env_args[args.environment].get("n_steps", 2048)
    expected_steps = math.ceil(target_steps / (n_steps * n_envs)) * n_steps * n_envs
    record = {
        "method": args.method,
        "seed": args.seed,
        "environment": environment_name,
        "environment_key": args.environment,
        "device": args.device,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "target_timesteps": target_steps,
        "expected_completed_timesteps": expected_steps,
        "cats_alpha": method_args.get("cats_alpha"),
        "lpp_profile": "B" if args.method == "lpp" else None,
        "auxiliary_metrics_enabled": False,
        "timing_protocol_version": 3,
        "implementation_optimization": "shared_policy_context_v2",
        "cats_optimization": (
            "reuse_current_context_no_aux_sync_v2"
            if args.method == "cats"
            else None
        ),
        "lpp_optimization": (
            "reuse_filtered_observation_no_mid_checkpoints_v1"
            if args.method == "lpp"
            else None
        ),
        "config": repr(method_args),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    output_path = timing_dir / f"{args.method}_seed{args.seed}.json"
    start = time.perf_counter()
    try:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        train_function(
            args.seed,
            target_steps,
            str(save_dir) + "/",
            str(log_dir) + "/",
            make_env,
            env_args[args.environment],
            method_args,
            device=args.device,
            detail_info=False,
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        record.update(
            {
                "status": "completed",
                "elapsed_seconds": elapsed,
                "elapsed_hours": elapsed / 3600,
                "end_to_end_fps": expected_steps / elapsed,
            }
        )
    except Exception as error:
        elapsed = time.perf_counter() - start
        record.update(
            {
                "status": "failed",
                "elapsed_seconds": elapsed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
