"""Real-time drowsy-driver EAR detector (webcam + dlib + OpenCV + pygame)."""

import os
import subprocess
import sys

import cv2
import dlib
import imutils
from imutils import face_utils
from pygame import mixer
from scipy.spatial import distance

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ALERT_SOUND = "music.wav"

# Alert when EAR stays below this threshold for frame_check consecutive frames.
thresh = 0.25
frame_check = 20

mixer.init()
_alert_loaded = False
_afplay_proc = None
if os.path.exists(ALERT_SOUND):
    mixer.music.load(ALERT_SOUND)
    _alert_loaded = True
else:
    print(f"Warning: '{ALERT_SOUND}' not found; visual alerts will still work.")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def play_alert():
    """Play the drowsiness alert via pygame, with macOS afplay fallback.

    Only starts audio when nothing is already playing, so a sustained
    closed-eye streak does not spawn overlapping afplay processes.
    """
    global _afplay_proc

    if _alert_loaded:
        try:
            if not mixer.music.get_busy():
                mixer.music.play()
            return
        except Exception as e:
            print(f"pygame alert failed ({e}); trying system audio…")

    if not (os.path.exists(ALERT_SOUND) and sys.platform == "darwin"):
        return
    if _afplay_proc is not None and _afplay_proc.poll() is None:
        return
    _afplay_proc = subprocess.Popen(
        ["afplay", ALERT_SOUND],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if not os.path.exists(PREDICTOR_PATH):
    print(
        f"Error: '{PREDICTOR_PATH}' not found.\n"
        "Run:  python setup_live_demo.py\n"
        "Or download https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 ,\n"
        "decompress it, and place the .dat file in the project root."
    )
    sys.exit(1)

try:
    predict = dlib.shape_predictor(PREDICTOR_PATH)
    detect = dlib.get_frontal_face_detector()
except Exception as e:
    print(f"Error loading dlib shape predictor: {e}")
    sys.exit(1)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(
        "Error: could not open webcam (camera index 0).\n"
        "On macOS: System Settings → Privacy & Security → Camera,\n"
        "and allow Terminal (or your IDE) camera access, then re-run."
    )
    sys.exit(1)

flag = 0
print(
    "Driver Drowsiness Detection running — look at the camera, "
    "close your eyes ~1s to trigger an alert. Press q to quit."
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(
                    frame,
                    "DROWSINESS ALERT!",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                play_alert()
        else:
            flag = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
