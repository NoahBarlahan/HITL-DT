"""Persistent fixture datum and deliberate scans from a repeatable camera pose."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from vision.apriltag_tracking import MM_PER_INCH, rotation_matrix_to_euler_degrees
from vision.vision_config import (
    APRILTAG_DICTIONARY, APRILTAG_SIZE_MM, TARGET_TAG_ID, FIXTURE_DATUM_FILE,
    FIXTURE_POSITION_TOLERANCE_IN, FIXTURE_ROTATION_TOLERANCE_DEG,
    FIXTURE_SAMPLE_COUNT, FIXTURE_MAX_REPROJECTION_PX,
    FIXTURE_STABILITY_POSITION_IN, FIXTURE_STABILITY_ROTATION_DEG,
)


def calibration_fingerprint(calibration):
    if calibration is None:
        return None
    digest = hashlib.sha256()
    for name in ('camera_matrix', 'distortion'):
        values = np.asarray(calibration[name], dtype='<f8')
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    digest.update(json.dumps(list(calibration['image_size'])).encode())
    return digest.hexdigest()


def rotation_angle_degrees(rotation):
    # atan2 is well behaved at both zero and 180 degrees.
    sine = np.linalg.norm([rotation[2, 1]-rotation[1, 2],
                           rotation[0, 2]-rotation[2, 0],
                           rotation[1, 0]-rotation[0, 1]]) / 2
    cosine = np.clip((np.trace(rotation)-1)/2, -1, 1)
    return float(np.degrees(np.arctan2(sine, cosine)))


def average_poses(samples):
    translations = np.array([p['translation_mm'] for p in samples])
    rotations = np.array([cv2.Rodrigues(p['rotation_vector'])[0] for p in samples])
    u, _, vt = np.linalg.svd(rotations.mean(axis=0))
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    translation = translations.mean(axis=0)
    position_spread = float(np.max(np.linalg.norm(translations-translation, axis=1)))
    rotation_spread = max(rotation_angle_degrees(rotation.T @ r) for r in rotations)
    return translation, rotation, position_spread, rotation_spread


def compare_pose(datum, translation, rotation):
    reference_rotation = np.asarray(datum['rotation_matrix'], dtype=float)
    # Delta is in the saved fixture tag axes, never the current camera axes.
    delta_mm = reference_rotation.T @ (translation-np.asarray(datum['translation_mm']))
    relative_rotation = reference_rotation.T @ rotation
    distance_in = float(np.linalg.norm(delta_mm)/MM_PER_INCH)
    angle_deg = rotation_angle_degrees(relative_rotation)
    moved = (distance_in > FIXTURE_POSITION_TOLERANCE_IN or
             angle_deg > FIXTURE_ROTATION_TOLERANCE_DEG)
    return dict(delta_in=delta_mm/MM_PER_INCH, distance_in=distance_in,
                angle_deg=angle_deg, moved=moved,
                delta_rpy_deg=rotation_matrix_to_euler_degrees(relative_rotation))


def load_datum(path):
    with Path(path).open(encoding='utf-8') as stream:
        data = json.load(stream)
    if data.get('version') != 1:
        raise ValueError('Unsupported datum format')
    for field in ('tag_id', 'tag_family', 'tag_size_mm', 'camera_identity',
                  'image_size', 'calibration_fingerprint', 'saved_at'):
        if field not in data:
            raise ValueError(f'Datum missing {field}')
    if not isinstance(data['tag_id'], int) or data['tag_id'] < 0:
        raise ValueError('Invalid datum tag ID')
    translation = np.asarray(data['translation_mm'], dtype=float)
    rotation = np.asarray(data['rotation_matrix'], dtype=float)
    if translation.shape != (3,) or rotation.shape != (3,3):
        raise ValueError('Invalid datum pose dimensions')
    if not np.isfinite(translation).all() or not np.isfinite(rotation).all() or translation[2] <= 0:
        raise ValueError('Invalid datum pose values')
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('Invalid datum rotation')
    return data


def save_datum(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    try:
        with temp.open('w', encoding='utf-8') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
        if path.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            shutil.copy2(path, path.with_name(f'{path.stem}.backup-{stamp}.json'))
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


class FixtureDatumSession:
    def __init__(self, camera_identity, calibration, path=FIXTURE_DATUM_FILE):
        self.path = Path(path)
        self.camera_identity = camera_identity
        self.calibration = calibration
        self.fingerprint = calibration_fingerprint(calibration)
        self.datum = None
        self.issue = None
        self.message = ''
        self.mode = 'save'
        self.samples = []
        self.capture_tag_id = None
        self.result = None
        self.live_samples = []
        self.live_message = ""
        self.started_at = None
        self.state = 'return'
        if self.path.exists():
            self.state = 'choice'
            try:
                self.datum = load_datum(self.path)
                self.issue = self.compatibility_issue()
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.issue = f'Cannot use saved datum: {error}'

    def compatibility_issue(self):
        d = self.datum
        if self.fingerprint is None:
            return 'Metric calibration unavailable; cannot compare datum'
        if d['camera_identity'] != self.camera_identity:
            return 'Camera changed; establish a new datum'
        if d['calibration_fingerprint'] != self.fingerprint:
            return 'Calibration changed; establish a new datum'
        if d['tag_family'] != int(APRILTAG_DICTIONARY) or d['tag_size_mm'] != APRILTAG_SIZE_MM:
            return 'Tag family / size changed; establish a new datum'
        if TARGET_TAG_ID is not None and d['tag_id'] != TARGET_TAG_ID:
            return 'Configured tag ID differs from datum'
        return None

    @property
    def expected_tag_id(self):
        if self.mode == 'compare' and self.datum:
            return self.datum['tag_id']
        return TARGET_TAG_ID

    def handle_key(self, key):
        if key == ord('k') and self.state in ('choice', 'return') and self.datum and not self.issue:
            self.mode, self.state = 'compare', 'return'
            self.message = 'Keeping saved datum.'
        elif key == ord('r') and self.state != 'capturing':
            self.state, self.mode = 'return', 'save'
            self.message = 'Replace datum only after a successful new capture.'
            self.result = None
        elif key == ord('p') and self.state == 'return':
            self.state = 'ready'
            self.message = 'Known camera pose confirmed. Hold camera and fixture still.'
        elif key == ord('s') and self.state == 'ready':
            if self.fingerprint is None:
                self.message = 'Cannot scan: metric camera calibration is required.'
            else:
                self.samples = []
                self.capture_tag_id = None
                self.started_at = time.monotonic()
                self.state = 'capturing'
                self.message = 'Collecting stable consecutive tag poses...'
        elif key == ord('c') and self.state == 'result' and self.datum:
            self.mode, self.state = 'compare', 'return'
            self.result = None
            self.message = 'Return the camera to the same known pose before checking.'
        elif key == ord('b') and self.state in ('ready', 'capturing', 'result'):
            self.samples = []
            self.live_samples = []
            self.result = None
            self.state = 'return'
            if self.datum and not self.issue:
                self.mode = 'compare'
            self.message = 'Comparison paused. Confirm the camera position again.'

    def observe(self, observations, calibration_ok=True):
        if self.state == 'result':
            self.update_live(observations, calibration_ok)
            return
        if self.state != 'capturing':
            return
        if time.monotonic()-self.started_at > 10:
            self.state = 'return'
            self.samples = []
            self.message = 'Scan timed out: hold still, improve tag visibility, and retry.'
            return
        eligible = [(tag, pose) for tag, pose in observations
                    if self.expected_tag_id is None or tag == self.expected_tag_id]
        if not calibration_ok:
            reason = 'Calibration unavailable or stream size changed.'
        elif len(eligible) != 1:
            reason = 'Show exactly one eligible fixture tag.' if self.expected_tag_id is None else f'Need one visible tag ID {self.expected_tag_id}.'
        else:
            tag, pose = eligible[0]
            reason = None
            if pose is None:
                reason = 'Metric tag pose unavailable.'
            elif (not np.isfinite(pose['reprojection_error']) or
                  pose['reprojection_error'] > FIXTURE_MAX_REPROJECTION_PX):
                reason = 'Tag fit is poor; improve lighting, focus or viewing angle.'
            elif (not np.isfinite(pose['translation_mm']).all() or
                  not np.isfinite(pose['rotation_vector']).all()):
                reason = 'Invalid tag pose.'
            elif self.capture_tag_id is not None and tag != self.capture_tag_id:
                reason = 'Tag ID changed during scan.'
        if reason:
            self.samples = []
            self.capture_tag_id = None
            self.message = reason
            return
        self.capture_tag_id = tag
        self.samples.append(pose)
        translation, rotation, spread_mm, spread_deg = average_poses(self.samples)
        if spread_mm/MM_PER_INCH > FIXTURE_STABILITY_POSITION_IN or spread_deg > FIXTURE_STABILITY_ROTATION_DEG:
            self.samples = []
            self.message = 'Pose unstable: camera/fixture motion or tag pose ambiguity. Retry while still.'
            return
        self.message = f'Stable samples {len(self.samples)}/{FIXTURE_SAMPLE_COUNT} | spread {spread_mm/MM_PER_INCH:.3f} in / {spread_deg:.2f} deg'
        if len(self.samples) < FIXTURE_SAMPLE_COUNT:
            return
        if self.mode == 'save':
            data = dict(version=1, tag_id=tag, tag_family=int(APRILTAG_DICTIONARY),
                        tag_size_mm=APRILTAG_SIZE_MM, camera_identity=self.camera_identity,
                        image_size=list(self.calibration['image_size']),
                        calibration_fingerprint=self.fingerprint,
                        translation_mm=translation.tolist(), rotation_matrix=rotation.tolist(),
                        saved_at=datetime.now(timezone.utc).isoformat(),
                        sample_count=len(self.samples), position_spread_mm=spread_mm,
                        rotation_spread_deg=spread_deg)
            try:
                save_datum(self.path, data)
            except OSError as error:
                self.state = 'return'
                self.message = f'Datum save failed: {error}'
                return
            self.datum, self.issue = data, None
            self.result = None
            self.message = 'Datum saved. This fixture pose is now the correct position.'
        else:
            self.result = compare_pose(self.datum, translation, rotation)
            self.message = 'FIXTURE MOVED' if self.result['moved'] else 'WITHIN TOLERANCE'
        self.mode = 'compare'
        self.live_samples = []
        self.state = 'result'
        self.update_live(observations, calibration_ok)

    def update_live(self, observations, calibration_ok=True):
        eligible = [(tag, pose) for tag, pose in observations
                    if self.datum and tag == self.datum['tag_id']]
        self.live_message = ''
        if not calibration_ok or self.issue or not self.datum:
            self.live_message = 'Error unavailable: camera calibration / datum mismatch.'
        elif len(eligible) != 1:
            self.live_message = f"Error unavailable: need visible tag {self.datum['tag_id']}."
        else:
            pose = eligible[0][1]
            if (pose is None or not np.isfinite(pose['reprojection_error']) or
                pose['reprojection_error'] > FIXTURE_MAX_REPROJECTION_PX or
                not np.isfinite(pose['translation_mm']).all() or
                not np.isfinite(pose['rotation_vector']).all()):
                self.live_message = 'Error unavailable: tag pose is unreliable.'
        if self.live_message:
            self.result = None
            self.live_samples = []
            return
        # A short rolling average reduces jitter without requiring another keypress.
        self.live_samples.append(pose)
        self.live_samples = self.live_samples[-5:]
        translation, rotation, _, _ = average_poses(self.live_samples)
        self.result = compare_pose(self.datum, translation, rotation)

    def dashboard_lines(self):
        if self.state == 'result':
            lines = [('LIVE ERROR FROM SAVED CORRECT POSITION', 'heading')]
            if self.result:
                x, y, z = self.result['delta_in']
                roll, pitch, yaw = self.result['delta_rpy_deg']
                lines += [
                    (f'X={x:+.3f} in    Y={y:+.3f} in    Z={z:+.3f} in', 'heading'),
                    (f'Roll={roll:+.2f} deg  Pitch={pitch:+.2f} deg  Yaw={yaw:+.2f} deg', 'heading'),
                    (f'Total position error: {self.result["distance_in"]:.3f} in | rotation: {self.result["angle_deg"]:.2f} deg', 'normal'),
                ]
            else:
                lines.append((self.live_message or 'Error unavailable: waiting for tag.', 'warning'))
            lines.append((f'Datum saved: tag {self.datum["tag_id"]} | offsets in saved tag axes', 'good'))
            lines.append(('Keep camera at known scan pose | [B] Pause | [R] Redo datum | [Q] Quit', 'normal'))
            return lines
        lines = [('FIXTURE DATUM | camera returns to the SAME known robot pose', 'heading'),
                 ('Camera position is confirmed manually; this app does not move the robot.', 'normal')]
        if self.state == 'choice':
            lines.append(('Startup: keep the correct position or establish a new one?', 'heading'))
            if self.datum and not self.issue:
                lines.append((f'[K] Keep saved tag {self.datum["tag_id"]} datum | [R] Redo correct position', 'normal'))
            else:
                lines.append(('[R] Capture a new correct position | [Q] Quit', 'normal'))
            if self.issue:
                lines.append((self.issue, 'warning'))
        elif self.state == 'return':
            label = 'SAVE DATUM' if self.mode == 'save' else 'CHECK FIXTURE'
            lines.append((f'{label}: return camera to your known scan pose, then press [P].', 'heading'))
            if self.mode == 'save':
                lines.append(('Place fixture in its correct position before saving. Old datum stays until save.', 'normal'))
                if self.datum and not self.issue:
                    lines.append(('[K] Keep previous datum instead', 'normal'))
        elif self.state == 'ready':
            action = 'save correct fixture datum' if self.mode == 'save' else 'check fixture against datum'
            lines.append((f'[S] Scan and {action} | [B] Back', 'heading'))
            if self.mode == 'save':
                lines.append(('P confirms the camera pose. S captures and SAVES the fixture datum.', 'normal'))
        elif self.state == 'capturing':
            lines.append(('Scanning: keep camera and fixture still | [B] Cancel', 'heading'))
        if self.state != 'result' and self.message:
            lines.append((self.message, 'warning'))
        lines.append((f'Movement limits: {FIXTURE_POSITION_TOLERANCE_IN:.3f} in total shift / {FIXTURE_ROTATION_TOLERANCE_DEG:.2f} deg rotation', 'normal'))
        if self.datum:
            lines.append((f'Datum tag {self.datum["tag_id"]} | saved {self.datum["saved_at"]}', 'normal'))
        lines.append(('Camera return error appears as fixture motion. Confirm pose before every scan.', 'warning'))
        return lines
