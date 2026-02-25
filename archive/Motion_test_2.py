import cv2
import numpy as np
import socket
import time

ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 1234

def send_cmd(cmd: str, timeout=3.0) -> str:
    with socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=timeout) as s:
        s.sendall(cmd.encode("utf-8"))
        s.shutdown(socket.SHUT_WR)
        s.settimeout(timeout)
        try:
            data = s.recv(4096)
            return data.decode(errors="replace").strip()
        except socket.timeout:
            return ""

def classify_color_hsv(hsv_roi) -> str:
    red1 = cv2.inRange(hsv_roi, (0, 120, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv_roi, (170, 120, 70), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    blue = cv2.inRange(hsv_roi, (100, 120, 70), (130, 255, 255))
    yellow = cv2.inRange(hsv_roi, (20, 120, 70), (35, 255, 255))

    counts = {
        "red": int(cv2.countNonZero(red)),
        "blue": int(cv2.countNonZero(blue)),
        "yellow": int(cv2.countNonZero(yellow)),
    }

    label, best = max(counts.items(), key=lambda kv: kv[1])
    total = hsv_roi.shape[0] * hsv_roi.shape[1]
    if best < 0.08 * total:
        return "none"
    return label

def main():
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (try camera index 0->1).")

    last_triggered = None
    last_time = 0.0
    debounce_s = 0.8

    print("Press 'q' in the video window (or ESC) to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        roi_size = int(min(h, w) * 0.25)
        x1 = w // 2 - roi_size // 2
        y1 = h // 2 - roi_size // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        color = classify_color_hsv(hsv)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"Detected: {color}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Adept Color Control", frame)

        now = time.time()
        if color != "none" and (color != last_triggered or (now - last_time) > debounce_s):
            last_triggered = color
            last_time = now

            try:
                if color == "red":
                    print("RED -> moveP1")
                    reply = send_cmd("moveP1")
                    if reply:
                        print("Reply:", reply)

                elif color == "blue":
                    print("BLUE -> moveP2")
                    reply = send_cmd("moveP2")
                    if reply:
                        print("Reply:", reply)

                elif color == "yellow":
                    # Ask the user what to do
                    while True:
                        choice = input(
                            "YELLOW detected. Choose: "
                            "[1] Move to Point 1, [2] Move to Point 2, [S] Stop, [Enter] Skip: "
                        ).strip().lower()

                        if choice == "":
                            print("Skipped.")
                            break
                        if choice == "1":
                            reply = send_cmd("moveP1")
                            if reply:
                                print("Reply:", reply)
                            break
                        if choice == "2":
                            reply = send_cmd("moveP2")
                            if reply:
                                print("Reply:", reply)
                            break
                        if choice == "s":
                            reply = send_cmd("stop")
                            if reply:
                                print("Reply:", reply)
                            else:
                                print("Sent stop.")
                            break

                        print("Invalid input. Type 1, 2, S, or Enter.")

            except Exception as e:
                print("Command failed:", e)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
