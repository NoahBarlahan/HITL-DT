import cv2
import numpy as np

OUT_FILE = r"C:\Users\noah\Desktop\Python Reference\HITLDT\homography.npy"

FRAME_W = 640
FRAME_H = 480
CAMERA_MAX_INDEX = 5


def open_camera(max_index: int = 5) -> cv2.VideoCapture:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for backend in backends:
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"Opened camera index {idx} using backend {backend}")
                return cap
            cap.release()
    raise RuntimeError(f"Could not open any webcam (tried indices 0-{max_index-1}).")


clicked_pixels = []  # [(u,v), ...]


def on_mouse(event, x, y, flags, param):
    global clicked_pixels
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_pixels.append((float(x), float(y)))
        print(f"Clicked pixel: ({x}, {y})")


def main():
    cap = open_camera(CAMERA_MAX_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Calibrate Homography")
    cv2.setMouseCallback("Calibrate Homography", on_mouse)

    print("\n=== HOMOGRAPHY CALIBRATION ===")
    print("Goal: map pixel (u,v) -> BoardFrame (Xmm,Ymm)")
    print("\nINSTRUCTIONS:")
    print("1) Make sure the board is visible and not moving.")
    print("2) Click 4 points on the board in THIS ORDER:")
    print("   P1: Board origin (0,0) or your chosen reference")
    print("   P2: +X direction point (X increases)")
    print("   P3: +X,+Y corner point")
    print("   P4: +Y direction point (Y increases)")
    print("\nAfter 4 clicks, you'll type the matching BoardFrame coordinates in mm.")
    print("Press R to reset clicks. Press Q to quit.\n")

    frozen = None

    while True:
        if frozen is None:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break
        else:
            frame = frozen.copy()

        vis = frame.copy()

        # Draw clicked points
        for i, (u, v) in enumerate(clicked_pixels):
            cv2.circle(vis, (int(u), int(v)), 6, (0, 0, 255), -1)
            cv2.putText(vis, f"P{i+1}", (int(u) + 8, int(v) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(vis, "Click 4 points. R=reset  S=freeze  Q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Calibrate Homography", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break

        if key in (ord('r'), ord('R')):
            clicked_pixels.clear()
            frozen = None
            print("Reset clicks.")

        if key in (ord('s'), ord('S')):
            # Freeze/unfreeze the image so the clicks match a stable frame
            frozen = None if frozen is not None else frame.copy()
            print("Frozen." if frozen is not None else "Unfrozen.")

        if len(clicked_pixels) == 4:
            # Freeze for safety
            frozen = frame.copy()

            print("\nYou clicked 4 pixel points:")
            for i, (u, v) in enumerate(clicked_pixels, start=1):
                print(f"  P{i} pixel: ({u:.1f}, {v:.1f})")

            print("\nNow enter the matching BoardFrame coordinates (mm) for each point.")
            print("Example: 0 0  (for origin)\n")

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
                        print("Invalid input. Please enter two numbers like: 120.5 30")

            pixel_pts = np.array(clicked_pixels, dtype=np.float32)
            board_pts = np.array(board_pts, dtype=np.float32)

            # Compute homography: pixel -> board
            H, status = cv2.findHomography(pixel_pts, board_pts, method=0)
            if H is None:
                print("ERROR: Could not compute homography.")
            else:
                np.save(OUT_FILE, H.astype(np.float64))
                print("\nSUCCESS! Saved homography to:")
                print(" ", OUT_FILE)
                print("\nHomography matrix H (pixel->board):\n", H)

                # Quick test: transform the 4 pixel points and show result
                pts = pixel_pts.reshape(-1, 1, 2)
                xy = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                print("\nCheck: transformed pixel points -> board (should match what you typed):")
                for i in range(4):
                    print(f"  P{i+1} -> ({xy[i,0]:.2f}, {xy[i,1]:.2f}) mm")

            print("\nDone. Press R to redo, or Q to quit.\n")

            # Prevent immediate re-trigger
            clicked_pixels.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()