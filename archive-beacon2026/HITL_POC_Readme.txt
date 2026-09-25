# HITL Edge-Based Color Sorting Pipeline
Cal Poly Manufacturing & Robotics Lab

## What It Does
Detects colored blocks via edge-based computer vision, classifies them
using k-means color clustering, and routes uncertain decisions to a human
operator. All decisions are logged to CSV for research analysis.

## Quick Start
  pip install opencv-python numpy
  python hitl_edge_combined.py

## Phase 1 — Training (~3 seconds)
Place all 3 color blocks clearly in frame, separated from each other.
The system auto-detects them and learns 3 color clusters. No input needed.

## Phase 2 — Inference
Each new block that enters frame is classified automatically if confidence
is high, or triggers an operator prompt if uncertain.

## Keys
  1 / 2 / 3   Assign block to BIN_1 / BIN_2 / BIN_3 (during HITL prompt)
  S           Skip — no assignment
  R           Restart training phase
  C           Print HSV of centre pixel to console
  Q / ESC     Quit and show session summary

## Configuration (top of file)
  CAMERA_INDEX        Webcam index — try 0 if camera does not open
  ROBOT_IP / PORT     TCP address of robot controller or emulator
  CONFIDENCE_THRESH   Below this (default 0.60) → HITL triggered
  OPERATOR_ID         Change per participant — written to all CSV rows
  TRAIN_SECONDS       Duration of training phase (default 3.0s)

## CSV Log  →  hitl_logs/session_<operator>_<timestamp>.csv
  timestamp, trial, color_detected, top_prob, second_color, second_prob,
  hitl_triggered, operator_override, final_bin, response_time_ms,
  hsv_H, hsv_S, hsv_V

## Robot TCP
  Sends plain ASCII commands over TCP. Default map:
    BIN_1 → moveP1    BIN_2 → moveP2    BIN_3 → moveP3
  Test without a robot: run  nc -lk 1234  to see commands print to terminal.
  If commands are not received, change CMD_TERMINATOR from "\n" to "\r\n".

## Troubleshooting
  Camera won't open          →  change CAMERA_INDEX to 0
  Training keeps restarting  →  blocks too similar in color; separate them
                                more physically, or check lighting
  Blocks detected but not    →  decrease STABLE_FRAMES or increase
  classified                    MISSING_FRAMES_TO_REMOVE
  Too many false detections  →  increase MIN_AREA_PX, use plain background

## Requirements
  Python 3.9+
  opencv-python
  numpy