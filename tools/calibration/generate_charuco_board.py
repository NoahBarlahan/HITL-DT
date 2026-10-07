"""Generate the ChArUco board used by calibrate_charuco.py.

Steps:
1. Edit the ChArUco settings in vision/vision_config.py if needed.
2. Run: py -3.12 -m tools.calibration.generate_charuco_board
3. Print the generated PNG at 100% scale, without "fit to page".
4. Measure the printed square size. If it differs from the configured size,
   update CHARUCO_SQUARE_LENGTH_MM and CHARUCO_MARKER_LENGTH_MM before running
   calibrate_charuco.py.
"""

import cv2

from vision.vision_config import (
    CHARUCO_DICTIONARY,
    CHARUCO_MARKER_LENGTH_MM,
    CHARUCO_OUTPUT_PIXELS_PER_SQUARE,
    CHARUCO_SQUARE_LENGTH_MM,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
    PROJECT_DIRECTORY,
)


OUTPUT_PATH = PROJECT_DIRECTORY / "calibration" / "charuco_board.png"
MARGIN_PIXELS = 40


def main():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Install opencv-contrib-python from requirements.txt.")

    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH_MM,
        CHARUCO_MARKER_LENGTH_MM,
        dictionary,
    )
    image_size = (
        CHARUCO_SQUARES_X * CHARUCO_OUTPUT_PIXELS_PER_SQUARE,
        CHARUCO_SQUARES_Y * CHARUCO_OUTPUT_PIXELS_PER_SQUARE,
    )
    image = board.generateImage(image_size, marginSize=MARGIN_PIXELS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT_PATH), image):
        raise RuntimeError(f"Could not write {OUTPUT_PATH}.")

    print(f"Wrote {OUTPUT_PATH}")
    print(
        "Print at 100% scale. Configured square size: "
        f"{CHARUCO_SQUARE_LENGTH_MM:.2f} mm."
    )


if __name__ == "__main__":
    main()
