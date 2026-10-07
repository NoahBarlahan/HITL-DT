"""Checks for persisted references, scan gating and rigid-pose comparisons."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from vision import fixture_datum as f


def calibration():
    return dict(camera_matrix=np.array([[600.,0,320],[0,600,240],[0,0,1]]),
                distortion=np.zeros(5), image_size=(640,480), camera_identity='test')


def pose(translation=(0,0,500), degrees=0):
    return dict(translation_mm=np.array(translation,dtype=float),
                rotation_vector=np.array([[0.],[0.],[np.radians(degrees)]]),
                reprojection_error=0.1)


class FixtureDatumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'datum.json'
        self.cal = calibration()

    def session(self, cal=None):
        return f.FixtureDatumSession('test',self.cal if cal is None else cal,self.path)

    def scan(self, session, value=None, tag=7):
        session.handle_key(ord('p'))
        session.handle_key(ord('s'))
        for _ in range(f.FIXTURE_SAMPLE_COUNT):
            session.observe([(tag,pose() if value is None else value)])

    def save_reference(self):
        session = self.session()
        self.scan(session)
        self.assertEqual(session.state,'result')
        self.assertTrue(self.path.exists())
        return session

    def test_first_boot_requires_position_confirmation(self):
        session = self.session()
        session.handle_key(ord('s'))
        session.observe([(7,pose())])
        self.assertEqual(session.state,'return')
        self.assertFalse(self.path.exists())
        self.scan(session)
        self.assertEqual(f.load_datum(self.path)['tag_id'],7)

    def test_restart_keep_same_pose_within_tolerance(self):
        self.save_reference()
        session = self.session()
        self.assertEqual(session.state,'choice')
        session.handle_key(ord('k'))
        self.scan(session)
        self.assertFalse(session.result['moved'])
        self.assertAlmostEqual(session.result['distance_in'],0)
        session.handle_key(ord('s'))
        self.assertEqual(session.state,'result')
        session.handle_key(ord('c'))
        self.assertEqual(session.state,'return')

    def test_translation_and_rotation_can_independently_trigger_motion(self):
        self.save_reference()
        for value in (pose((3,0,500)),pose(degrees=2)):
            with self.subTest(value=value):
                session = self.session()
                session.handle_key(ord('k'))
                self.scan(session,value)
                self.assertTrue(session.result['moved'])

    def test_translation_delta_uses_saved_tag_axes(self):
        reference = dict(translation_mm=[0,0,500],rotation_matrix=cv2.Rodrigues(pose(degrees=90)['rotation_vector'])[0].tolist())
        rotation = np.array(reference['rotation_matrix'])
        result = f.compare_pose(reference,np.array([25.4,0,500]),rotation)
        np.testing.assert_allclose(result['delta_in'],[0,-1,0],atol=1e-10)
        self.assertAlmostEqual(result['angle_deg'],0)
        self.assertAlmostEqual(f.rotation_angle_degrees(cv2.Rodrigues(pose(degrees=180)['rotation_vector'])[0]),180)

    def test_bad_fit_missing_tag_wrong_tag_reset_capture(self):
        self.save_reference()
        for observations in ([],[(8,pose())],[(7,None)],[(7,dict(pose(),reprojection_error=5))]):
            session = self.session()
            session.handle_key(ord('k'))
            session.handle_key(ord('p'))
            session.handle_key(ord('s'))
            session.observe([(7,pose())])
            session.observe(observations)
            self.assertEqual(session.samples,[])
            self.assertEqual(session.state,'capturing')

    def test_multiple_tags_cannot_become_arbitrary_first_datum(self):
        session = self.session()
        session.handle_key(ord('p'))
        session.handle_key(ord('s'))
        session.observe([(7,pose()),(8,pose())])
        self.assertEqual(session.samples,[])
        self.assertFalse(self.path.exists())

    def test_unstable_capture_not_saved(self):
        session = self.session()
        session.handle_key(ord('p'))
        session.handle_key(ord('s'))
        session.observe([(7,pose())])
        session.observe([(7,pose((10,0,500)))])
        self.assertEqual(session.samples,[])
        self.assertFalse(self.path.exists())

    def test_redo_preserves_old_datum_until_success_and_backs_it_up(self):
        self.save_reference()
        old = self.path.read_text()
        session = self.session()
        session.handle_key(ord('r'))
        self.assertEqual(self.path.read_text(),old)
        self.scan(session,pose((3,0,500)))
        self.assertNotEqual(self.path.read_text(),old)
        backups = list(self.path.parent.glob('datum.backup-*.json'))
        self.assertEqual(len(backups),1)
        self.assertEqual(backups[0].read_text(),old)

    def test_changed_camera_calibration_tag_or_corrupt_file_blocks_keep(self):
        self.save_reference()
        cal = copy.deepcopy(self.cal)
        cal['camera_matrix'][0,0] += 1
        session = self.session(cal)
        session.handle_key(ord('k'))
        self.assertEqual(session.state,'choice')
        with patch.object(f,'APRILTAG_SIZE_MM',100):
            session = self.session()
            self.assertIsNotNone(session.issue)
        session = f.FixtureDatumSession('different-camera',self.cal,self.path)
        session.handle_key(ord('k'))
        self.assertEqual(session.state,'choice')
        self.path.write_text('{broken')
        session = self.session()
        self.assertIsNotNone(session.issue)
        session.handle_key(ord('r'))
        self.scan(session)
        self.assertTrue(self.path.exists())

    def test_timeout_cancel_and_missing_calibration(self):
        session = self.session()
        session.handle_key(ord('p'))
        session.handle_key(ord('s'))
        session.started_at -= 11
        session.observe([(7,pose())])
        self.assertEqual(session.state,'return')
        session.handle_key(ord('p'))
        session.handle_key(ord('s'))
        session.handle_key(ord('b'))
        self.assertEqual(session.state,'return')
        session = f.FixtureDatumSession('test',None,self.path)
        session.handle_key(ord('p'))
        session.handle_key(ord('s'))
        self.assertEqual(session.state,'ready')
        self.assertFalse(self.path.exists())

    def test_live_offsets_update_without_another_scan(self):
        session = self.save_reference()
        original = self.path.read_text()
        for _ in range(5):
            session.observe([(7,pose((2.54,0,500)))])
        self.assertAlmostEqual(session.result['delta_in'][0],0.1)
        self.assertEqual(self.path.read_text(),original)
        self.assertTrue(any('LIVE ERROR' in line for line,_ in session.dashboard_lines()))

    def test_live_tag_loss_bad_pose_and_recovery_do_not_show_stale_errors(self):
        session = self.save_reference()
        for observations in ([],[(8,pose())],[(7,None)],[(7,dict(pose(),reprojection_error=5))]):
            session.observe(observations)
            self.assertIsNone(session.result)
            self.assertTrue(any('unavailable' in line for line,_ in session.dashboard_lines()))
            session.observe([(7,pose())])
            self.assertIsNotNone(session.result)
        session.observe([(7,pose())],calibration_ok=False)
        self.assertIsNone(session.result)

    def test_pause_disables_live_comparison_until_reconfirmed(self):
        session = self.save_reference()
        session.handle_key(ord('b'))
        session.observe([(7,pose((10,0,500)))])
        self.assertEqual(session.state,'return')
        self.assertIsNone(session.result)
        self.assertEqual(session.mode,'compare')

    def test_save_failure_preserves_reference(self):
        self.save_reference()
        old = self.path.read_text()
        session = self.session()
        session.handle_key(ord('r'))
        with patch.object(f,'save_datum',side_effect=OSError('test failure')):
            self.scan(session,pose((3,0,500)))
        self.assertEqual(self.path.read_text(),old)
        self.assertEqual(session.state,'return')


if __name__ == '__main__':
    unittest.main()
