import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile

spec = importlib.util.spec_from_file_location('package', Path(__file__).with_name('package.py'))
package = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader and Path(spec.origin).exists():
    spec.loader.exec_module(package)


class ContractTests(unittest.TestCase):
    def test_route_receipt_requires_go_and_no_hidden_gt(self):
        good = {
            'completed': True,
            'comparison_decision': 'GO',
            'hidden_ground_truth_used': False,
            'rows': 1000,
            'candidate_rows': 4000,
            'selection_counts': {'1': 380, '2': 128, '3': 181, '4': 311},
        }
        package.validate_route_receipt(good)
        for key, bad_value in [
            ('completed', False),
            ('comparison_decision', 'NO_GO'),
            ('hidden_ground_truth_used', True),
            ('rows', 999),
            ('candidate_rows', 3999),
        ]:
            bad = dict(good)
            bad[key] = bad_value
            with self.assertRaises(ValueError):
                package.validate_route_receipt(bad)

    def test_plan_preserves_varied_source_paths_and_caps(self):
        self.assertTrue(hasattr(package, 'make_plan'), 'missing frozen-path planning')
        rows = [{'episode_id': 1, 'selected_seed': 3, 'selected_video': '/safe/shard0/episode1.mp4', 'selected_video_sha256': 'a'*64},
                {'episode_id': 2, 'selected_seed': 1, 'selected_video': '/safe/old/episode_000002.mp4', 'selected_video_sha256': 'b'*64}]
        result = package.make_plan(rows, {1: 98, 2: 487}, Path('/safe'), count=2)
        self.assertEqual([(r['episode_id'], r['expected_frames'], r['needs_trim']) for r in result], [(1,98,True),(2,121,False)])
        self.assertEqual(result[1]['selected_video'], '/safe/old/episode_000002.mp4')
        with self.assertRaises(ValueError):
            package.make_plan(rows + rows[:1], {1:98,2:487}, Path('/safe'), count=2)
        rows[0]['selected_video'] = '/outside/episode1.mp4'
        with self.assertRaises(ValueError):
            package.make_plan(rows, {1:98,2:487}, Path('/safe'), count=2)

    def test_trim_does_not_resample_and_limits_threads(self):
        self.assertTrue(hasattr(package, 'trim_command'), 'missing native-FPS trim')
        cmd = package.trim_command('ffmpeg', Path('/in.mp4'), Path('/out.mp4'), 98)
        self.assertEqual(cmd[cmd.index('-frames:v')+1], '98')
        self.assertEqual(cmd[cmd.index('-fps_mode')+1], 'passthrough')
        self.assertNotIn('-r', cmd)
        self.assertNotIn('-vf', cmd)
        self.assertEqual([cmd[i+1] for i,x in enumerate(cmd) if x == '-threads'], ['2','2'])
        self.assertIn('-n', cmd)

    def test_archive_rejects_duplicate_members_and_mismatch(self):
        self.assertTrue(hasattr(package, 'verify_archive'), 'missing strict ZIP verification')
        root = Path(tempfile.mkdtemp(prefix='wa2-package-test-'))
        archive = root / 'test.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('README.md', b'identity')
            z.writestr('HZ-World/episode1.mp4', b'video')
        import hashlib
        expected = {'README.md': hashlib.sha256(b'identity').hexdigest(), 'HZ-World/episode1.mp4': hashlib.sha256(b'video').hexdigest()}
        self.assertEqual(package.verify_archive(archive, expected), 2)
        with self.assertRaises(ValueError):
            package.verify_archive(archive, {'README.md': 'bad'})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(archive, 'a') as z:
                z.writestr('README.md', b'duplicate')
        with self.assertRaises(ValueError):
            package.verify_archive(archive, expected)


if __name__ == '__main__':
    unittest.main()
