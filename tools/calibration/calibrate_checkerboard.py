"""Calibrate the D435IF COLOR camera using your Calib.io 15 mm checkerboard.

STEPS:
1. Keep the board rigid and planar. Tilt is useful; bending/bowing is not.
2. Count all squares across and down. Run with those actual counts:
       py -3.12 -m tools.calibration.calibrate_checkerboard --squares ACROSS DOWN
   For example ONLY, 10x7 squares means --squares 10 7 (9x6 inner corners).
   Alternatively configure CHECKERBOARD_COLUMNS/ROWS as INNER corner counts.
3. Use the same resolution, sensor, and focus for calibration and detection.
   This script uses the RealSense COLOR sensor through vision/camera_source.py.
   Close the depth viewer/other camera applications before starting.
4. Press C for a stationary, sharp, fully visible board view. Move/tilt between
   captures. Cover center, all four edges, near/far working distances, tilts
   about both axes, and in-plane rotations. Prefer 25-40 diverse images.
5. Press Enter to compute after >=15 views. There is NO automatic stop at 30:
   quality/diversity matter more than image count. U undoes the last capture.
6. Review printed per-view RMS pixel errors. High-error views are candidates
   for review, not automatically deleted. Capture new views if necessary.
7. After solving, press V to toggle undistorted preview; use S to explicitly
   save. Existing calibration is backed up before replacement. C adds another
   view and invalidates the previous solution. Q exits without saving.
8. Run CV-AprilTagModel.py with the measured 77.8 mm detection-edge tag size.
   Validate distance with a ruler at several distances/angles and image edges.
   The 100 mm outer paper size is not the pose-estimation tag size.

Captured images are preserved in timestamped calibration/sessions folders.
This is lens calibration, not depth-sensor or robot hand-eye calibration.
"""

import argparse
from datetime import datetime
import json
import shutil

import cv2
import numpy as np

from vision.camera_source import open_camera
from vision.vision_config import (CALIBRATION_FILE, PROJECT_DIRECTORY,
                           CHECKERBOARD_COLUMNS, CHECKERBOARD_ROWS,
                           CHECKERBOARD_SQUARE_MM)

MIN_VIEWS = 15


def object_points(columns, rows, square_mm):
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_mm
    return points


def solve_calibration(points, image_points, image_size):
    result = cv2.calibrateCameraExtended(
        [points.copy() for _ in image_points], image_points, image_size, None, None)
    rms, matrix, distortion, _, _, deviations, _, view_errors = result
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("Calibration produced non-finite values; collect better views")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Invalid focal length")
    print(f"Overall RMS: {rms:.3f} px (not a millimeter accuracy guarantee)")
    for index, error in enumerate(view_errors.flatten(), 1):
        print(f"  View {index:02d}: {error:.3f} px")
    print("Review the largest errors; verify actual distances after saving.")
    return dict(camera_matrix=matrix, distortion_coefficients=distortion,
                reprojection_error=rms, per_view_errors=view_errors,
                intrinsic_standard_deviations=deviations)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--squares", type=int, nargs=2, metavar=("ACROSS", "DOWN"))
    parser.add_argument("--square-mm", type=float, default=CHECKERBOARD_SQUARE_MM)
    args = parser.parse_args()
    columns, rows = CHECKERBOARD_COLUMNS, CHECKERBOARD_ROWS
    if args.squares:
        columns, rows = (value - 1 for value in args.squares)
    if columns is None or rows is None:
        parser.error("Supply --squares ACROSS DOWN using your board's actual square counts")
    if columns < 3 or rows < 3 or args.square_mm <= 0:
        parser.error("Need >=3 inner corners each direction and positive square size")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session = PROJECT_DIRECTORY / "calibration" / "sessions" / stamp
    session.mkdir(parents=True)
    points = object_points(columns, rows, args.square_mm)
    views, filenames = [], []
    solution = None
    preview = False
    coverage = set()
    capture = open_camera()
    image_size = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera frame unavailable")
            size = (frame.shape[1], frame.shape[0])
            if image_size is not None and size != image_size:
                raise RuntimeError("Stream size changed; restart calibration")
            image_size = size
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(
                gray, (columns, rows), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
            display = frame.copy()
            if preview and solution is not None:
                display = cv2.undistort(display, solution["camera_matrix"],
                                        solution["distortion_coefficients"])
            elif found:
                cv2.drawChessboardCorners(display, (columns, rows), corners, found)
            status = f"Views {len(views)} | board {'found' if found else 'not found'} | coverage {len(coverage)}/9 cells"
            cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0) if found else (0, 0, 255), 1)
            cv2.putText(display, "C capture U undo Enter solve S save V preview Q quit",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Checkerboard calibration", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                if not found:
                    print("Capture rejected: entire configured checkerboard must be detected")
                    continue
                if any(np.mean(np.linalg.norm(corners.reshape(-1, 2) - old.reshape(-1, 2), axis=1)) < 8
                       for old in views):
                    print("Capture rejected: too similar to an existing view; change position/tilt")
                    continue
                path = session / f"view_{len(filenames)+1:03d}_{datetime.now().strftime('%H%M%S_%f')}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError("Could not preserve capture image")
                views.append(corners.astype(np.float32))
                filenames.append(str(path))
                solution = None
                preview = False
                coverage = {(min(2, max(0, int(x / size[0] * 3))),
                             min(2, max(0, int(y / size[1] * 3))))
                            for view in views for x, y in view.reshape(-1, 2)}
                print(f"Captured {len(views)}; coverage {len(coverage)}/9. Aim for 25-40 varied views.")
            if key == ord("u") and views:
                views.pop()
                filenames.pop()  # Preserved image remains on disk for inspection.
                solution = None
                preview = False
                coverage = {(min(2, max(0, int(x / size[0] * 3))),
                             min(2, max(0, int(y / size[1] * 3))))
                            for view in views for x, y in view.reshape(-1, 2)}
            if key in (10, 13):
                if len(views) < MIN_VIEWS:
                    print(f"Need at least {MIN_VIEWS} varied views")
                    continue
                solution = solve_calibration(points, views, image_size)
                print("Press V to preview undistortion, S to save, or C to add more views")
            if key == ord("v") and solution is not None:
                preview = not preview
            if key == ord("s"):
                if solution is None:
                    print("Press Enter to compute calibration before saving")
                    continue
                metadata = dict(image_width=size[0], image_height=size[1],
                                camera_identity=capture.camera_identity, capture_count=len(views),
                                checkerboard_columns=columns, checkerboard_rows=rows,
                                square_length_mm=args.square_mm,
                                created_at=datetime.now().astimezone().isoformat())
                if CALIBRATION_FILE.exists():
                    backup = CALIBRATION_FILE.with_name(f"camera_calibration_backup_{stamp}.npz")
                    shutil.copy2(CALIBRATION_FILE, backup)
                    print(f"Previous calibration backed up: {backup}")
                np.savez(CALIBRATION_FILE, **solution, **metadata)
                report = {**metadata, "images": filenames,
                          "rms_pixels": float(solution["reprojection_error"]),
                          "per_view_rms_pixels": solution["per_view_errors"].flatten().tolist()}
                (session / "report.json").write_text(json.dumps(report, indent=2))
                print(f"Saved {CALIBRATION_FILE}; report/images in {session}")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
