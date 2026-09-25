import os
import cv2
import PySpin
import time

SAVE_DIR = "dataset"

CLASSES = {
    "p": "perfect",
    "u": "usable",
    "x": "unusable",
}

CLASS_DIRS = {
    name: os.path.join(SAVE_DIR, name)
    for name in CLASSES.values()
}

for d in CLASS_DIRS.values():
    os.makedirs(d, exist_ok=True)


def init_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No FLIR camera found.")

    cam = cam_list.GetByIndex(0)
    cam.Init()

    nodemap = cam.GetNodeMap()

    try:
        acq_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        acq_continuous = acq_mode.GetEntryByName("Continuous")
        acq_mode.SetIntValue(acq_continuous.GetValue())
    except Exception:
        print("[WARN] Could not set acquisition mode.")

    cam.BeginAcquisition()

    processor = PySpin.ImageProcessor()

    try:
        processor.SetColorProcessing(PySpin.HQ_LINEAR)
    except Exception:
        pass

    return system, cam_list, cam, processor


def get_frame(cam, processor):
    image_result = cam.GetNextImage(1000)

    try:
        if image_result.IsIncomplete():
            return None

        try:
            converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
        except Exception:
            converted = image_result.Convert(PySpin.PixelFormat_BGR8)

        frame = converted.GetNDArray()

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return frame.copy()

    finally:
        image_result.Release()


def release_camera(system, cam_list, cam):
    try:
        cam.EndAcquisition()
        cam.DeInit()
        cam_list.Clear()
        system.ReleaseInstance()
    except Exception:
        pass


def count_images():
    counts = {}

    for class_name, class_dir in CLASS_DIRS.items():
        counts[class_name] = len([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    return counts


def main():
    system = None
    cam_list = None
    cam = None

    counts = count_images()

    print("\nControls:")
    print("p = save PERFECT tee")
    print("u = save USABLE but imperfect tee")
    print("x = save UNUSABLE tee")
    print("q = quit")
    print("\nRecommended: collect at least 50 images per class.\n")

    try:
        system, cam_list, cam, processor = init_camera()

        while True:
            frame = get_frame(cam, processor)

            if frame is None:
                continue

            display = frame.copy()

            y = 35
            for class_name, count in counts.items():
                cv2.putText(
                    display,
                    f"{class_name.upper()}: {count}",
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                y += 40

            cv2.putText(
                display,
                "p=PERFECT | u=USABLE | x=UNUSABLE | q=QUIT",
                (20, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Golf Tee Dataset Capture", display)

            key = cv2.waitKey(1) & 0xFF
            key_char = chr(key) if key < 128 else ""

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            if key_char in CLASSES:
                class_name = CLASSES[key_char]
                class_dir = CLASS_DIRS[class_name]
                count = counts[class_name]

                filename = os.path.join(
                    class_dir,
                    f"{class_name}_{count:04d}_{timestamp}.png"
                )

                cv2.imwrite(filename, frame)
                counts[class_name] += 1

                print(f"[SAVE] {class_name.upper()} -> {filename}")

            elif key_char == "q" or key == 27:
                break

    finally:
        cv2.destroyAllWindows()

        if cam is not None:
            release_camera(system, cam_list, cam)

        print("\nDone.")
        for class_name, count in counts.items():
            print(f"{class_name}: {count}")

        print(f"Saved in: {SAVE_DIR}")


if __name__ == "__main__":
    main()