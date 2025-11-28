import cv2
import mediapipe as mp
import math
import numpy as np
import threading

from gps.overlay import GPSOverlay  

# ============================
# 0) AXIS CAMERA STREAM
# ============================

STREAM_URL = "http://192.168.1.2/axis-cgi/mjpg/video.cgi?camera=1&resolution=2048x1536"

# ============================
# 1) Load calibration from gps_overlay.json
# ============================

overlay = GPSOverlay()

camera_matrix = np.array(overlay.data["camera_matrix"], dtype=np.float32)
dist_coeffs   = np.array(overlay.data["distortion_coeffs"], dtype=np.float32)
calib_size    = tuple(overlay.calib_size)      # (2048, 1536)
margin        = overlay.margin_pixels          # e.g. 200
expanded_size = tuple(overlay.corrected_size)  # e.g. (2448, 1936)

new_camera_matrix = camera_matrix.copy()
new_camera_matrix[0, 2] += margin
new_camera_matrix[1, 2] += margin

scale_factor = 0.8
new_camera_matrix[0, 0] *= scale_factor
new_camera_matrix[1, 1] *= scale_factor

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    camera_matrix,
    dist_coeffs,
    np.eye(3),
    new_camera_matrix,
    expanded_size,
    cv2.CV_16SC2
)

# ============================
# 2) MEDIAPIPE SETUP 
# ============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


# ============================
# 3) GESTURE LOGIC
# ============================

def is_palm_open(lm):
    wrist = lm[0]
    fingers = [
        (8, 5),   # index
        (12, 9),  # middle
        (16, 13), # ring
        (20, 17)  # pinky
    ]
    extended = 0
    for tip_id, mcp_id in fingers:
        tip = lm[tip_id]
        mcp = lm[mcp_id]
        d_tip_wrist = dist(tip, wrist)
        d_mcp_wrist = dist(mcp, wrist)

        if d_tip_wrist > d_mcp_wrist * 1.05:
            extended += 1

    return extended >= 2


def is_fist(lm):
    wrist = lm[0]
    fingers = [
        (8, 5),
        (12, 9),
        (16, 13),
        (20, 17)
    ]
    folded = 0
    for tip_id, mcp_id in fingers:
        tip = lm[tip_id]
        mcp = lm[mcp_id]
        d_tip_mcp = dist(tip, mcp)
        d_mcp_wrist = dist(mcp, wrist)
        d_tip_wrist = dist(tip, wrist)

        if d_tip_mcp < d_mcp_wrist * 0.8 and d_tip_wrist < d_mcp_wrist * 1.3:
            folded += 1

    return folded >= 2


def get_gesture_from_landmarks(lm):
    """
    Return 'FIST', 'PALM' or None
    """
    if is_fist(lm):
        return "FIST"
    if is_palm_open(lm):
        return "PALM"
    return None


def get_position(lm, frame_width, frame_height):
    """
    Return (x, y) in pixels as the center of the hand.
    lm = list of 21 landmarks with normalized coordinates [0,1]
    """
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]

    cx = int(sum(xs) / len(xs) * frame_width)
    cy = int(sum(ys) / len(ys) * frame_height)

    return (cx, cy)


# ============================
# 4) CLASS-BASED API
# ============================

class HandRecognizer:
    def __init__(self, stream_url: str = STREAM_URL):
        self.stream_url = stream_url

        self.cap = None
        self.running = False
        self.thread = None

        self._lock = threading.Lock()
        self._gesture = None        # 'FIST' / 'PALM' / None
        self._position = None       # (x, y) eller None

    def _loop(self):
        # Open camera
        self.cap = cv2.VideoCapture(self.stream_url)

        if not self.cap.isOpened():
            print("Failed to open Axis camera stream...")
            self.running = False
            return

        # Create MediaPipe Hands locally in the loop (no global "hands")
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,             # we only need one hand
            model_complexity=0,          # faster model
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2
        ) as hands:

            while self.running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame from Axis camera...")
                    continue

                # Ensure correct size for undistortion
                h, w = frame.shape[:2]
                if (w, h) != calib_size:
                    frame_scaled = cv2.resize(frame, calib_size, interpolation=cv2.INTER_LINEAR)
                else:
                    frame_scaled = frame

                # Undistortion
                undistorted = cv2.remap(frame_scaled, map1, map2, cv2.INTER_LINEAR)

                # === SMALL IMAGE FOR MEDIAPIPE (FASTER) ===
                # scale DOWN the image and use undistorted-width/height for position calculation
    
                small = cv2.resize(undistorted, (640, 480))
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                results = hands.process(small_rgb)

                gesture = None
                position = None
                text_color = (255, 255, 255)
                text_label = "NO HAND"

                if results.multi_hand_landmarks:
                    # Take first hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    lm = hand_landmarks.landmark

                    # Gesture
                    gesture = get_gesture_from_landmarks(lm)
                    if gesture == "FIST":
                        text_color = (0, 0, 255)
                        text_label = "FIST"
                    elif gesture == "PALM":
                        text_color = (0, 255, 0)
                        text_label = "PALM"
                    else:
                        text_color = (255, 255, 255)
                        text_label = "UNKNOWN"

                    # Position in PIXELS on the UNDISTORTED image
                    fh, fw = undistorted.shape[:2]
                    position = get_position(lm, fw, fh)

                    # Draw landmarks on undistorted (if you want to debug)
                    mp_drawing.draw_landmarks(
                        undistorted, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                # Update shared state
                with self._lock:
                    self._gesture = gesture
                    self._position = position

                # Draw text on the image
                cv2.putText(
                    undistorted,
                    text_label,
                    (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    text_color,
                    3
                )

                # Show debug window (can be commented out if running headless)
                cv2.imshow('Hand Recognition (Axis Camera, undistorted)', undistorted)

                # Close with 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break

        # Cleanup
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def run(self):
        """
        Start the camera loop in a background thread.
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        """
        Stop the background thread and close the camera/window.
        """
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def get_gesture(self):
        """
        Return the latest gesture:
        'FIST', 'PALM' or None if no hand/gesture currently.
        """
        with self._lock:
            return self._gesture

    def get_position(self):
        """
        Return the latest hand position as (x, y) in pixels,
        or None if no hand is detected.
        """
        with self._lock:
            return self._position
