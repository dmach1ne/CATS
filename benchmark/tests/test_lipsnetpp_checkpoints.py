import tempfile
import unittest
from pathlib import Path

from cats_common.lipsnetpp_experiment import run_name, selected_profile
from cats_common.lipsnetpp_noise_evaluation import discover_selected_checkpoints


class CheckpointDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seeds = [31, 2, 71, 9, 48, 5, 99, 12, 53, 6]

    def populate(self, algorithm):
        seed_file = self.root / algorithm / 'tests' / 'seeds.txt'
        seed_file.parent.mkdir(parents=True)
        seed_file.write_text('\n'.join(map(str, self.seeds + [123])))
        results = self.root / algorithm / 'results' / 'pths' / 'hopper'
        profile = selected_profile(algorithm, 'hopper')
        expected = [[], []]
        for seed in self.seeds:
            base = (results / f'vanilla_ppo_{seed}' / 'final.zip'
                    if algorithm == 'ppo' else results / f'vanilla_{seed}.zip')
            lipsnet = results / run_name(algorithm, profile, seed) / 'final.zip'
            for paths, path in zip(expected, (base, lipsnet)):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                paths.append(path)
        return expected

    def test_ppo_and_sac_follow_seed_order_and_selected_profile(self):
        for algorithm in ('ppo', 'sac'):
            with self.subTest(algorithm=algorithm):
                expected = self.populate(algorithm)
                pairs = discover_selected_checkpoints(algorithm, 'hopper', self.root)
                for checkpoints, paths in zip(pairs, expected):
                    self.assertEqual([c.training_seed for c in checkpoints], self.seeds)
                    self.assertEqual([c.path for c in checkpoints], paths)

    def test_missing_checkpoint_does_not_silently_reduce_seed_count(self):
        missing = self.populate('ppo')[1][-1]
        missing.unlink()
        with self.assertRaises(FileNotFoundError) as error:
            discover_selected_checkpoints('ppo', 'hopper', self.root)
        self.assertIn(str(missing), str(error.exception))

    def test_short_seed_file_is_rejected(self):
        self.populate('ppo')
        (self.root / 'ppo/tests/seeds.txt').write_text('31\n2\n')
        with self.assertRaises(ValueError):
            discover_selected_checkpoints('ppo', 'hopper', self.root)

    def test_duplicate_seeds_are_rejected(self):
        self.populate('ppo')
        (self.root / 'ppo/tests/seeds.txt').write_text('31\n' * 10)
        with self.assertRaises(ValueError):
            discover_selected_checkpoints('ppo', 'hopper', self.root)


if __name__ == '__main__':
    unittest.main()
