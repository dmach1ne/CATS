import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CatsLoggingTest(unittest.TestCase):
    def test_no_diagnostic_gradients_or_statistics(self):
        for algorithm in ('ppo', 'sac'):
            with self.subTest(algorithm=algorithm):
                path = ROOT / algorithm / 'models' / f'cats_{algorithm}.py'
                tree = ast.parse(path.read_text())
                calls = [ast.unparse(n.func) for n in ast.walk(tree)
                         if isinstance(n, ast.Call)]
                self.assertNotIn('th.autograd.grad', calls)
                self.assertNotIn('th.quantile', calls)
                self.assertFalse(any('_poc_stats' in c or 'loss_tail_stats' in c
                                     for c in calls))

    def test_temporal_loss_and_coefficient_are_still_logged(self):
        for algorithm in ('ppo', 'sac'):
            source = (ROOT / algorithm / 'models' / f'cats_{algorithm}.py').read_text()
            self.assertIn('"train/loss_t"', source)
            self.assertIn('"train/cats_lambda"', source)


if __name__ == '__main__':
    unittest.main()
