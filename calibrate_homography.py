import cv2
import numpy as np
from pathlib import Path
import PySpin

# Save homography beside this script
OUT_FILE = Path(__file__).resolve().parent / "homography.npy"

clicked_pixels = []  # [(u,v), ...]


# =========================================================
# FLIR / PYSPIN HELPERS
# =========================================================
def _set_enumeration_node(nodemap, node_name: str, entry_name: str) -> bool:
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        return False
    entry = node.GetEntryByName(entry_name)
    if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
        return False
    node.SetIntValue(entry.GetValue())
    return True


def _try_set_first_pixel_format(nodemap, candidates):
    for fmt in candidates:
        if _set_enumeration_node(nodemap, "PixelFormat", fmt):
            return fmt
    return None


def init_flir_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No FLIR cameras detected.")

    cam = cam_list.GetByIndex(0)
    cam.Init()
    nodemap = cam.GetNodeMap()

    _set_enumeration_node(nodemap, "AcquisitionMode", "Continuous")
    _try_set_first_pixel_format(
        nodemap,
        ["BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8", "RGB8", "BGR8", "Mono8"],
    )

    cam.BeginAcquisition()

    processor = PySpin.ImageProcessor()
    try:
        processor.SetColorProcessing(PySpin.HQ_LINEAR)
    except Exception:
        pass

    print("[INFO] FLIR camera ready.")
    return system, cam_list, cam, processor


def release_flir_camera(system, cam_list, cam):
    try:
        if cam:
            try:
                cam.EndAcquisition()
            except Exception:
                pass
            try:
                cam.DeInit()
            except Exception:
                pass
    finally:
        try:
            if cam_list:
                cam_list.Clear()
        except Exception:
            pass
        try:
            if system:
                system.ReleaseInstance()
        except Exception:
            pass


def get_bgr_frame(cam, processor):
    image_result = cam.GetNextImage(1000)
    try:
        if image_result.IsIncomplete():
            return None

        converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
        arr = converted.GetNDArray()
        if arr is None:
            return None

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
            if arr.shape[2] == 3:
                return arr.copy()
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        return None
    finally:
        image_result.Release()


# =========================================================
# UI
# =========================================================
def on_mouse(event, x, y, flags, param):
    global clicked_pixels
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixels.append((float(x), float(y)))
        print(f"Clicked pixel: ({x}, {y})")


def main():
    global clicked_pixels

    system = cam_list = cam = processor = None
    frozen = None

    try:
        system, cam_list, cam, processor = init_flir_camera()

        cv2.namedWindow("Calibrate Homography")
        cv2.setMouseCallback("Calibrate Homography", on_mouse)

        print("\n=== HOMOGRAPHY CALIBRATION ===")
        print("Goal: map pixel (u,v) -> BoardFrame (Xmm,Ymm)")
        print("\nINSTRUCTIONS:")
        print("1) Make sure the board is visible and not moving.")
        print("2) Press S to freeze when you have a good image.")
        print("3) Click 4 points on the board in THIS ORDER:")
        print("   P1: Board origin (0,0) or your chosen reference")
        print("   P2: +X direction point")
        print("   P3: +X,+Y corner point")
        print("   P4: +Y direction point")
        print("4) Enter the matching board coordinates in mm.")
        print("\nKeys: S=freeze/unfreeze, R=reset, Q=quit\n")

        while True:
            if frozen is None:
                frame = get_bgr_frame(cam, processor)
                if frame is None:
                    continue
            else:
                frame = frozen.copy()

            vis = frame.copy()

            for i, (u, v) in enumerate(clicked_pixels):
                cv2.circle(vis, (int(u), int(v)), 6, (0, 0, 255), -1)
                cv2.putText(
                    vis,
                    f"P{i+1}",
                    (int(u) + 8, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.putText(
                vis,
                "Click 4 points. S=freeze R=reset Q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Calibrate Homography", vis)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key in (ord("r"), ord("R")):
                clicked_pixels.clear()
                frozen = None
                print("Reset clicks.")

            if key in (ord("s"), ord("S")):
                frozen = None if frozen is not None else frame.copy()
                print("Frozen." if frozen is not None else "Unfrozen.")

            if len(clicked_pixels) == 4:
                frozen = frame.copy()

                print("\nYou clicked 4 pixel points:")
                for i, (u, v) in enumerate(clicked_pixels, start=1):
                    print(f"  P{i} pixel: ({u:.1f}, {v:.1f})")

                print("\nNow enter the matching BoardFrame coordinates (mm).")
                board_pts = []
                for i in range(4):
                    while True:
                        s = input(f"Enter Board mm for P{i+1} as 'X Y': ").strip()
                        try:
                            x_str, y_str = s.split()
                            X = float(x_str)
                            Y = float(y_str)
                            board_pts.append((X, Y))
                            break
                        except Exception:
                            print("Invalid input. Enter two numbers like: 120.5 30")

                pixel_pts = np.array(clicked_pixels, dtype=np.float32)
                board_pts = np.array(board_pts, dtype=np.float32)

                H, status = cv2.findHomography(pixel_pts, board_pts, method=0)
                if H is None:
                    print("ERROR: Could not compute homography.")
                else:
                    np.save(OUT_FILE, H.astype(np.float64))
                    print(f"\nSUCCESS! Saved homography to:\n  {OUT_FILE}")
                    print("\nHomography matrix H (pixel->board):\n", H)

                    pts = pixel_pts.reshape(-1, 1, 2)
                    xy = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

                    print("\nCheck: transformed pixel points -> board:")
                    for i in range(4):
                        print(f"  P{i+1} -> ({xy[i,0]:.2f}, {xy[i,1]:.2f}) mm")

                print("\nDone. Press R to redo, or Q to quit.\n")
                clicked_pixels.clear()

    finally:
        cv2.destroyAllWindows()
        release_flir_camera(system, cam_list, cam)


if __name__ == "__main__":
    main()