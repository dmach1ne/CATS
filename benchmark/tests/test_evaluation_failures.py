import ast
import csv
import io
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import Mock

import numpy as np

from cats_common.checkpoints import select_checkpoints


ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def test_select_requested_seeds_in_order(self):
        for files in (
            ["cats_3/final.zip", "cats_1/final.zip", "cats_2/final.zip"],
            ["vanilla_3.zip", "vanilla_1.zip", "vanilla_2.zip"],
        ):
            self.assertEqual(select_checkpoints(files, [2, 1]), [files[2], files[1]])

    def test_missing_seed_fails_even_with_same_count(self):
        with self.assertRaises(FileNotFoundError):
            select_checkpoints(["cats_1/final.zip", "cats_3/final.zip"], [1, 2])

    def test_empty_or_duplicate_seeds_fail(self):
        for seeds in ([], [1, 1]):
            with self.assertRaises(ValueError):
                select_checkpoints([], seeds)

    def test_duplicate_checkpoints_fail(self):
        with self.assertRaises(ValueError):
            select_checkpoints(["cats_1.zip", "cats_1/final.zip"], [1])


class EvaluationFailureTests(unittest.TestCase):
    def evaluator(self, algorithm):
        path = ROOT / algorithm / "tests/modules/action_extractor.py"
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        namespace = dict(
            __file__=str(path), os=os, Path=Path, np=np, traceback=traceback, csv=csv,
            max_loop=2,
            select_checkpoints=select_checkpoints, seed_file_path="unused",
            train_envs_dict={"hopper": Mock(side_effect=RuntimeError("environment failed"))},
        )
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        namespace["load_seeds"] = lambda _: [1, 2]
        return namespace

    def test_no_checkpoints_does_not_write_scores(self):
        for algorithm in ("ppo", "sac"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp:
                ns = self.evaluator(algorithm)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(FileNotFoundError):
                    ns["test_some_path"](
                        tmp + "/", add_al_names=["cats"], sub_dir="out",
                        target_envs=["hopper"], expected_seeds=[1, 2],
                    )
                self.assertFalse(list(Path(tmp).rglob("*.csv")))
                ns["train_envs_dict"]["hopper"].assert_not_called()

    def test_runtime_error_propagates(self):
        for algorithm in ("ppo", "sac"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / "hopper/cats_1/final.zip"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
                ns = self.evaluator(algorithm)
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaisesRegex(RuntimeError, "environment failed"):
                    ns["test_some_path"](
                        tmp + "/", add_al_names=["cats"], sub_dir="out",
                        target_envs=["hopper"], expected_seeds=[1],
                    )
                self.assertFalse(list(Path(tmp).rglob("*.csv")))

    def test_successful_evaluation_writes_measured_scores(self):
        for algorithm in ("ppo", "sac"):
            with self.subTest(algorithm=algorithm), tempfile.TemporaryDirectory() as tmp:
                for seed in (1, 2):
                    checkpoint = Path(tmp) / f"hopper/cats_{seed}/final.zip"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.touch()
                ns = self.evaluator(algorithm)
                env = Mock()
                env.reset.return_value = (np.zeros(3), {})
                env.step.return_value = (np.zeros(3), 2.0, True, False, {})
                ns["train_envs_dict"]["hopper"] = lambda mode: lambda: env
                model = Mock()
                model.predict.return_value = (np.zeros(1), None)
                ns["CustomPPO" if algorithm == "ppo" else "CustomSAC"] = Mock()
                ns["CustomPPO" if algorithm == "ppo" else "CustomSAC"].load.return_value = model
                with redirect_stdout(io.StringIO()):
                    ns["test_some_path"](
                        tmp + "/", add_al_names=["cats"], sub_dir="out",
                        target_envs=["hopper"], expected_seeds=[1, 2],
                    )
                with (Path(tmp) / "out/re_mean.csv").open() as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(float(rows[1][1]), 2.0)

    def test_training_worker_failures_are_reraised(self):
        for algorithm in ("ppo", "sac"):
            for name in ("train_cats", "train_fixed_controls", "train_cats_next_noise", "test_all"):
                path = ROOT / algorithm / "tests" / f"{name}.py"
                tree = ast.parse(path.read_text())
                handlers = [
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.ExceptHandler) and node.name == "exc"
                ]
                self.assertTrue(handlers, path)
                for handler in handlers:
                    self.assertTrue(any(isinstance(node, ast.Raise) for node in handler.body), path)


if __name__ == "__main__":
    unittest.main()
