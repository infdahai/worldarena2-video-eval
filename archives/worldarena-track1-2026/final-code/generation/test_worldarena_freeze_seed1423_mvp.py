import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class Seed3AppendTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('worldarena_freeze_seed1423_mvp.py')
        self.assertTrue(path.exists(), 'missing CPU seed3 append implementation')
        spec = importlib.util.spec_from_file_location('route1423', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def prior(self):
        return dict(episode_id=1, prior_seed=4, selected_seed=4,
                    seed1_mean9=.9, seed4_mean9=.6, seed2_mean9=.7,
                    selected_video='/prior.mp4', selected_video_sha256='prior')

    def test_compare_only_frozen_selected_seed_not_best_old_score(self):
        prior = self.prior()
        row = self.module.append_seed3(prior, .65, '/seed3.mp4', 'new')
        self.assertEqual(row['selected_seed'], 3)
        self.assertEqual(row['selected_video'], '/seed3.mp4')
        self.assertEqual(row['selected_video_sha256'], 'new')
        self.assertEqual(prior, self.prior())

    def test_tie_keeps_frozen_video_and_sha(self):
        row = self.module.append_seed3(self.prior(), .6, '/seed3.mp4', 'new')
        self.assertEqual(row['selected_seed'], 4)
        self.assertEqual(row['selected_video'], '/prior.mp4')
        self.assertEqual(row['selected_video_sha256'], 'prior')

    def test_weaker_seed3_does_not_reselect_other_old_seed(self):
        row = self.module.append_seed3(self.prior(), .5, '/seed3.mp4', 'new')
        self.assertEqual(row['selected_seed'], 4)

    def test_seed2_incumbent_can_be_replaced(self):
        prior = self.prior()
        prior['selected_seed'] = 2
        row = self.module.append_seed3(prior, .71, '/seed3.mp4', 'new')
        self.assertEqual(row['selected_seed'], 3)
        self.assertEqual(row['seed142_selected_seed'], 2)

    def test_invalid_seed_and_scores_fail_closed(self):
        for seed, incumbent, candidate in [(3, .6, .7), (4, float('nan'), .7),
                                            (4, .6, float('inf')), (4, .6, -1)]:
            row = self.prior()
            row['selected_seed'] = seed
            row[f'seed{seed}_mean9'] = incumbent
            with self.assertRaises(ValueError):
                self.module.append_seed3(row, candidate, '/new.mp4', 'new')

    def receipts(self, root, count=8):
        for shard in range(count):
            path = root / f'scoring/generated9/seed3/shard{shard}/package/receipts/test1000-generated9.complete.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(completed=True, seed=3, shard=shard,
                                            rows=125, hidden_ground_truth_used=False)))

    def test_missing_final_receipt_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.receipts(root, 7)
            with self.assertRaises(FileNotFoundError):
                self.module.require_complete_inputs(root)
            self.assertFalse((root / 'routing').exists())

    def test_incomplete_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.receipts(root)
            path = root / 'scoring/generated9/seed3/shard7/package/receipts/test1000-generated9.complete.json'
            data = json.loads(path.read_text())
            data['completed'] = False
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                self.module.require_complete_inputs(root)

    def test_accepts_all_eight_complete_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.receipts(root)
            self.assertEqual(len(self.module.require_complete_inputs(root)), 8)


if __name__ == '__main__':
    unittest.main()
