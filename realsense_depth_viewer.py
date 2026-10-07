"""Inspect camera depth pixels, filtering, and calibrated AprilTag pose on the D435IF.

Setup:
1. Connect the D435/D435i directly to a USB 3 port.
2. Install project packages: py -3.12 -m pip install -r requirements.txt
3. Run: py -3.12 realsense_depth_viewer.py
   To verify streaming without opening a window, add --check-streams.

Controls:
  Q or Esc  quit
  [ / ]     decrease / increase the far depth-display range
  R         reset the far depth-display range to 1.50 m
  S         toggle spatial smoothing
  T         toggle temporal smoothing (off by default when moving the camera)
  H         toggle hole filling (visual estimates; off by default)
  V         toggle raw / filtered depth image
  Click     select a 5x5 patch in either camera image

The saved checkerboard calibration is used ONLY for the RGB AprilTag pose.
RealSense's factory depth intrinsics and depth/color extrinsics still register
depth to color; RGB calibration does not recalibrate the stereo depth sensor.
Distance displays are inches; tag rotation is degrees.
Internal calibration and tag-size settings stay in mm, SDK depth stays meters.

The window shows color on the left and a colorized, depth-aligned image on the
right. Raw tag depth is sampled before filtering/filling in a 5x5 pixel patch.
The comparison transforms the tag pose into the DEPTH sensor frame first; raw
aligned depth encodes depth-sensor Z, not straight-line range. Registration and
patch sampling are approximate near boundaries or on slanted surfaces. These
measurements are shown separately; no depth/pose fusion is applied.
Use this script to inspect depth quality; it is not a calibrated robot-control
signal. Shiny, dark, occluded, or distant surfaces can produce invalid/noisy
depth and must be validated before use in a placement workflow.
"""

import argparse

import cv2
import numpy as np
import pyrealsense2 as rs
from vision.apriltag_tracking import (MM_PER_INCH, load_calibration, calibration_status,
                               estimate_tag_pose, tag_center_pixel)
from vision.vision_config import (COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS,
                           APRILTAG_DICTIONARY, APRILTAG_SIZE_MM,
                           CALIBRATION_FILE, TARGET_TAG_ID)


WIDTH = COLOR_WIDTH
HEIGHT = COLOR_HEIGHT
FRAMES_PER_SECOND = COLOR_FPS
NEAR_RANGE_M = 0.15
DEFAULT_FAR_RANGE_M = 1.50
PATCH_RADIUS_PIXELS = 2


def configure_streams():
    """Pin the device and enable camera streams only."""
    devices = rs.context().query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense camera found.")
    device = devices[0]
    config = rs.config()
    config.enable_device(device.get_info(rs.camera_info.serial_number))
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FRAMES_PER_SECOND)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FRAMES_PER_SECOND)
    return config


def patch_depth_mm(depth_frame, pixel):
    """Return a valid-pixel median and fraction in a clipped 5x5 patch."""
    center_x, center_y = (int(round(float(value))) for value in pixel)
    if not (0 <= center_x < depth_frame.get_width() and 0 <= center_y < depth_frame.get_height()):
        return None, 0.0
    depths_m = []
    total = 0
    for y in range(max(0, center_y - PATCH_RADIUS_PIXELS),
                   min(depth_frame.get_height(), center_y + PATCH_RADIUS_PIXELS + 1)):
        for x in range(max(0, center_x - PATCH_RADIUS_PIXELS),
                       min(depth_frame.get_width(), center_x + PATCH_RADIUS_PIXELS + 1)):
            total += 1
            distance_m = depth_frame.get_distance(x, y)
            if distance_m > 0:
                depths_m.append(distance_m)
    if not depths_m:
        return None, 0.0
    return float(np.median(depths_m) * 1000.0), len(depths_m) / total


def center_patch_depth_mm(depth_frame):
    return patch_depth_mm(depth_frame, (depth_frame.get_width() // 2,
                                       depth_frame.get_height() // 2))[0]


class CalibratedTagOverlay:
    def __init__(self, profile):
        self.calibration = load_calibration()
        identity = "realsense:" + profile.get_device().get_info(rs.camera_info.serial_number)
        self.issue = calibration_status(self.calibration, identity, (WIDTH, HEIGHT))
        if self.issue:
            self.calibration = None
            print(self.issue)
        else:
            print(f"Using saved RGB calibration: {CALIBRATION_FILE}")
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY), params)
        color = profile.get_stream(rs.stream.color)
        depth = profile.get_stream(rs.stream.depth)
        self.color_to_depth = color.get_extrinsics_to(depth)
        self.detected_frames = 0
        self.pose_frames = 0

    def draw(self, color_image, depth_display, raw_aligned_depth):
        corners, ids, _ = self.detector.detectMarkers(
            cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY))
        status = self.issue or f"RGB calibration active | tag edge {APRILTAG_SIZE_MM / MM_PER_INCH:.3f} in"
        if ids is None:
            return [status, "No AprilTag detected"]
        ids = np.asarray(ids).reshape(-1, 1)
        cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
        eligible = [i for i, tag_id in enumerate(ids.flatten())
                    if TARGET_TAG_ID is None or int(tag_id) == TARGET_TAG_ID]
        if not eligible:
            return [status, f"Target tag {TARGET_TAG_ID} not visible"]
        index = min(eligible, key=lambda i: int(ids[i, 0]))
        self.detected_frames += 1
        tag_id = int(ids[index, 0])
        pixel = tag_center_pixel(corners[index])
        center = tuple(np.rint(pixel).astype(int))
        for image in (color_image, depth_display):
            cv2.drawMarker(image, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        measured_mm, valid = patch_depth_mm(raw_aligned_depth, pixel)
        measured_text = "invalid" if measured_mm is None else f"{measured_mm / MM_PER_INCH:.2f} in"
        lines = [status, f"Tag {tag_id}: RAW sensor Z at tag={measured_text}, valid patch={valid:.0%}"]
        pose = estimate_tag_pose(corners[index], self.calibration)
        if pose is None:
            return lines + ["Tag metric pose unavailable"]
        self.pose_frames += 1
        x, y, z = pose["translation_mm"] / MM_PER_INCH
        distance = float(np.linalg.norm(pose["translation_mm"]) / MM_PER_INCH)
        roll, pitch, yaw = pose["euler_degrees"]
        # Compare in the native DEPTH frame, accounting for the RGB/depth offset.
        predicted_depth_point = rs.rs2_transform_point_to_point(
            self.color_to_depth, (pose["translation_mm"] / 1000.0).tolist())
        expected_mm = predicted_depth_point[2] * 1000.0
        difference = "invalid" if measured_mm is None else f"{(measured_mm - expected_mm) / MM_PER_INCH:+.3f} in"
        lines += [f"Tag in RGB frame: X={x:+.2f} Y={y:+.2f} Z={z:+.2f} in | range={distance:.2f} in",
                  f"Tag R/P/Y={roll:+.1f}/{pitch:+.1f}/{yaw:+.1f} deg | reprojection={pose['reprojection_error']:.2f} px",
                  f"Depth - tag expected Z: {difference} (depth frame, approximate patch comparison)"]
        return lines


def colorize_depth(depth_image, far_range_m):
    """Map depth in the selected display range to a readable false-color image."""
    far_range_mm = far_range_m * 1000.0
    near_range_mm = NEAR_RANGE_M * 1000.0
    depth_mm = depth_image.astype(np.float32)
    normalized = np.clip(
        (depth_mm - near_range_mm) / (far_range_mm - near_range_mm), 0.0, 1.0
    )
    colored = cv2.applyColorMap(
        (255.0 * (1.0 - normalized)).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored[(depth_mm > 0) & ((depth_mm < near_range_mm) | (depth_mm > far_range_mm))] = (65, 65, 65)
    colored[depth_mm <= 0] = (100, 0, 150)
    return colored



def pixel_patch(depth_mm, pixel):
    """Keep the exact sampled pixels; NaN represents positions outside the image."""
    x, y = (int(round(float(v))) for v in pixel)
    r = PATCH_RADIUS_PIXELS
    patch = np.full((2*r+1, 2*r+1), np.nan, dtype=np.float32)
    x0, x1 = max(0, x-r), min(depth_mm.shape[1], x+r+1)
    y0, y1 = max(0, y-r), min(depth_mm.shape[0], y+r+1)
    if x1 > x0 and y1 > y0:
        patch[y0-y+r:y1-y+r, x0-x+r:x1-x+r] = depth_mm[y0:y1, x0:x1]
    return patch


def patch_summary(patch):
    present = np.isfinite(patch)
    valid = patch[present & (patch > 0)]
    count = int(present.sum())
    median = None if not valid.size else float(np.median(valid))
    span = None if not valid.size else float(np.ptp(valid))
    return median, span, int(valid.size), count


def put_text(image, text, x, y, scale=0.48, color=(225, 230, 235)):
    # Fit dashboard labels without clipping when the configured RGB width changes.
    width = max(1, image.shape[1] - x - 12)
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    scale = min(scale, scale * width / max(1, text_width))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def draw_pixel_grid(image, patch, raw_patch, origin, cell, far_range_m, title):
    x0, y0 = origin
    put_text(image, title, x0, y0 - 12, 0.52)
    colors = colorize_depth(np.nan_to_num(patch), far_range_m)
    for row in range(patch.shape[0]):
        for col in range(patch.shape[1]):
            x, y = x0 + col*cell, y0 + row*cell
            value = patch[row, col]
            color = tuple(int(c) for c in colors[row, col])
            cv2.rectangle(image, (x, y), (x+cell-1, y+cell-1), color, -1)
            cv2.rectangle(image, (x, y), (x+cell-1, y+cell-1), (25, 25, 25), 1)
            label = 'N/A' if not np.isfinite(value) else ('X' if value <= 0 else f'{value / MM_PER_INCH:.2f}')
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            pos = (x+(cell-size[0])//2, y+(cell+size[1])//2)
            cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
            if np.isfinite(value) and value > 0 and raw_patch[row, col] == 0:
                cv2.rectangle(image, (x+2, y+2), (x+cell-3, y+cell-3), (255, 255, 0), 2)
            if row == PATCH_RADIUS_PIXELS and col == PATCH_RADIUS_PIXELS:
                cv2.rectangle(image, (x+5, y+5), (x+cell-6, y+cell-6), (255, 255, 255), 1)


def dashboard(color_image, raw_mm, filtered_mm, pixel, far_range_m,
              use_spatial, use_temporal, use_holes, show_raw, tag_lines,
              native_invalid, sensor_status):
    """Render exact raw/filtered aligned pixels and explain filter tradeoffs."""
    header = 64
    footer = 500
    panel = np.full((HEIGHT+header+footer, WIDTH*2, 3), (23, 26, 31), dtype=np.uint8)
    raw_patch = pixel_patch(raw_mm, pixel)
    filtered_patch = pixel_patch(filtered_mm, pixel)
    raw_median, raw_span, raw_valid, total = patch_summary(raw_patch)
    filt_median, filt_span, filt_valid, _ = patch_summary(filtered_patch)
    fmt = lambda v: 'invalid' if v is None else f'{v / MM_PER_INCH:.3f} in'
    put_text(panel, f'CAMERA | 5x5 median: raw {fmt(raw_median)} | filtered {fmt(filt_median)}', 12, 25, 0.62)
    put_text(panel, 'Click image: select patch | V: raw/filtered view | [ / ]: display range | R: reset range | Q: quit', 12, 50)
    depth_display = colorize_depth(raw_mm if show_raw else filtered_mm, far_range_m)
    rgb = color_image.copy()
    center = tuple(int(v) for v in pixel)
    r = PATCH_RADIUS_PIXELS
    for image in (rgb, depth_display):
        cv2.rectangle(image, (center[0]-r, center[1]-r), (center[0]+r, center[1]+r), (255, 255, 255), 1)
        cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 18, 1)
    put_text(rgb, 'RGB | white box = sampled 5x5 patch', 10, 23)
    put_text(depth_display, f'{"RAW" if show_raw else "FILTERED"} aligned depth | {NEAR_RANGE_M / 0.0254:.1f}-{far_range_m / 0.0254:.1f} in', 10, 23)
    panel[header:header+HEIGHT, :WIDTH] = rgb
    panel[header:header+HEIGHT, WIDTH:] = depth_display
    left = panel[header+HEIGHT:, :WIDTH]
    lines = [
        (f'[S] Spatial smoothing: {"ON" if use_spatial else "OFF"}', True),
        ('+ Reduces pixel speckle by smoothing neighboring depths.', False),
        ('- Can soften edges and erase small or shallow features.', False),
        (f'[T] Temporal smoothing: {"ON" if use_temporal else "OFF"}', True),
        ('+ Reduces frame-to-frame flicker when the scene is still.', False),
        ('- Can lag or smear moving objects. Missing pixels stay missing.', False),
        (f'[H] Hole filling: {"ON" if use_holes else "OFF"}', True),
        ('+ Covers gaps using neighboring depths for a fuller picture.', False),
        ('- Estimates missing values; can hide gaps and distort geometry.', False),
        ('5x5 median: rejects isolated outliers; may miss tiny features.', False),
        ('Invalid depth = no stereo match; distance is only one cause.', True),
        ('Shiny / dark / plain surfaces, exposure and occlusion cause gaps.', False),
        ('Try a matte textured target; check IR exposure / emitter in Viewer.', False),
        ('RGB checkerboard calibration does not calibrate stereo depth.', False),
    ]
    for index, (line, strong) in enumerate(lines):
        put_text(left, line, 12, 23+index*22, 0.48,
                 (130, 215, 255) if strong else (210, 215, 220))
    # Keep pose details separate from filter help and pixel diagnostics.
    for index, line in enumerate(tag_lines):
        put_text(left, line, 12, 337+index*18, 0.36)
    right = panel[header+HEIGHT:, WIDTH:]
    missing_aligned = float(np.mean(raw_mm <= 0))
    put_text(right, f'Missing: native {native_invalid:.1%} | RGB-aligned {missing_aligned:.1%}', 12, 23, 0.50)
    put_text(right, sensor_status, 12, 46, 0.41)
    cell = min(58, (WIDTH-42)//10)
    draw_pixel_grid(right, raw_patch, raw_patch, (12, 83), cell, far_range_m, 'RAW pixels (in)')
    draw_pixel_grid(right, filtered_patch, raw_patch, (WIDTH//2+8, 83), cell, far_range_m, 'FILTERED pixels (in)')
    y = 83+cell*5
    put_text(right, f'Raw valid {raw_valid}/{total} | span {fmt(raw_span)}', 12, y+23, 0.43)
    put_text(right, f'Filtered valid {filt_valid}/{total} | span {fmt(filt_span)}', 12, y+45, 0.43)
    put_text(right, 'Magenta / X: missing | gray: valid, outside color range', 12, y+67, 0.42)
    put_text(right, 'Cyan border: raw missing, now estimated | white: center pixel', 12, y+88, 0.42)
    put_text(right, 'Aligned pixels may repeat native samples. Span is spatial spread, not accuracy.', 12, y+109, 0.36)
    return panel


def sensor_description(sensor):
    parts = []
    for option, label in ((rs.option.enable_auto_exposure, 'AE'),
                          (rs.option.exposure, 'exposure us'),
                          (rs.option.gain, 'gain'),
                          (rs.option.emitter_enabled, 'emitter')):
        if sensor.supports(option):
            parts.append(f'{label}={sensor.get_option(option):g}')
    return f'{WIDTH}x{HEIGHT} @ {FRAMES_PER_SECOND}fps | ' + ' | '.join(parts)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-streams", action="store_true",
                        help="Verify 60 depth/color frames without a window.")
    args = parser.parse_args()
    pipeline = rs.pipeline()
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 0)  # Keep missing values visible.
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.holes_fill, 0)  # Do not retain missing pixels.
    holes = rs.hole_filling_filter()
    to_disparity = rs.disparity_transform(True)
    to_depth = rs.disparity_transform(False)
    use_spatial, use_temporal, use_holes = True, False, False

    started = False
    far_range_m = DEFAULT_FAR_RANGE_M
    selected_pixel = [WIDTH // 2, HEIGHT // 2]
    show_raw = False
    window = "RealSense depth viewer"

    def select_patch(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and 64 <= y < 64+HEIGHT and 0 <= x < WIDTH*2:
            selected_pixel[:] = [x % WIDTH, y-64]

    try:
        config = configure_streams()
        profile = pipeline.start(config)
        started = True
        device_name = profile.get_device().get_info(rs.camera_info.name)
        sensor = profile.get_device().first_depth_sensor()
        depth_scale_m = sensor.get_depth_scale()
        print(f"Connected to {device_name}")
        print("Depth is aligned to color. Press Q or Esc to exit.")
        tags = CalibratedTagOverlay(profile)

        align_to_color = rs.align(rs.stream.color)
        checked_frames = 0
        if not args.check_streams:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, select_patch)
        while True:
            frames = pipeline.wait_for_frames(5000)
            native = np.asanyarray(frames.get_depth_frame().get_data())
            native_invalid = float(np.mean(native == 0))
            aligned_frames = align_to_color.process(frames)
            aligned_depth = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not aligned_depth or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data()).copy()
            raw_aligned_depth = aligned_depth
            raw_mm = np.asanyarray(aligned_depth.get_data()).astype(np.float32) * depth_scale_m * 1000.0
            processed = to_disparity.process(aligned_depth)
            if use_spatial:
                processed = spatial.process(processed)
            if use_temporal:
                processed = temporal.process(processed)
            processed = to_depth.process(processed)
            if use_holes:
                processed = holes.process(processed)
            aligned_depth = processed.as_depth_frame()
            depth_image = np.asanyarray(aligned_depth.get_data())
            filtered_mm = depth_image.astype(np.float32) * depth_scale_m * 1000.0
            depth_display = colorize_depth(filtered_mm, far_range_m)
            tag_lines = tags.draw(color_image, depth_display, raw_aligned_depth)

            if args.check_streams:
                checked_frames += 1
                if checked_frames >= 60:
                    print(f"Stream check passed: {checked_frames} aligned, filtered depth/color frames.")
                    print(f"Tag detections: {tags.detected_frames}; calibrated poses: {tags.pose_frames}")
                    for line in tag_lines:
                        print(line)
                    print(f"Native missing depth: {native_invalid:.1%}")
                    print(sensor_description(sensor))
                    break
                continue

            panel = dashboard(color_image, raw_mm, filtered_mm, selected_pixel,
                              far_range_m, use_spatial, use_temporal, use_holes,
                              show_raw, tag_lines, native_invalid, sensor_description(sensor))
            cv2.imshow(window, panel)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("["):
                far_range_m = max(NEAR_RANGE_M + 0.10, far_range_m - 0.10)
            if key == ord("]"):
                far_range_m += 0.10
            if key == ord("r"):
                far_range_m = DEFAULT_FAR_RANGE_M
            if key == ord("s"):
                use_spatial = not use_spatial
            if key == ord("t"):
                use_temporal = not use_temporal
                temporal = rs.temporal_filter()  # Clear historical samples.
                temporal.set_option(rs.option.holes_fill, 0)
            if key == ord("h"):
                use_holes = not use_holes
            if key == ord("v"):
                show_raw = not show_raw
    except RuntimeError as error:
        print(f"RealSense error: {error}")
        print("Close other camera applications; check the USB cable and requested stream profiles.")
        if args.check_streams:
            raise
    finally:
        if started:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
