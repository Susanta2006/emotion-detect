import sys
import threading
from datetime import datetime
import cv2
from deepface import DeepFace
import pyfiglet
import time
import tensorflow
TF_ENABLE_ONEDNN_OPTS=0
# Banner
pf = pyfiglet.figlet_format("Emotion Detector")
print(pf, "\n version 2.1 (Threaded, Fixed Webcam)")
print('''
**********************************
* ------------------------------ *
* |Created by Mr. Susanta Banik| *
* ------------------------------ *
**********************************
''')
print("------------------------------------------------------- :INSTRUCTIONS: ----------------------------------------------------------------------------------")
print('''[?]General Instructions:
   ********************

(1) Look At the camera/webcam properly..
(2) Clean The Webcam for better outputs..
(3) Press 'q' to exit...
''')
print("------------------------------------------------------- :Data Processing: ---------------------------------------------------------------------------------")
print()

# Shared resources
frame_buffer = None
latest_result = {}
result_lock = threading.Lock()
running = True

# Emotion detection thread
def analyze_emotion():
    global frame_buffer, latest_result, running
    while running:
        if frame_buffer is None:
            time.sleep(0.01)
            continue
        try:
            resized = cv2.resize(frame_buffer, (1200, 1060))
            result = DeepFace.analyze(resized, actions=['emotion', 'age', 'gender', 'race'], enforce_detection=False)
            with result_lock:
                latest_result = result[0]
        except Exception:
            with result_lock:
                latest_result = {}
        time.sleep(0.01)  # Wait a second before next analysis

# Initialize webcam once
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[!] Error: Could not open webcam.")
    sys.exit()

# Set camera resolution to 1280x720 (HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Start thread
thread = threading.Thread(target=analyze_emotion)
thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame.")
            break

        # Update shared frame buffer
        frame_buffer = frame.copy()

        # Get latest analysis results
        with result_lock:
            result = latest_result.copy()

        if result:
            overlay_text = (
                f"Emotion: {result.get('dominant_emotion', '')} | "
                f"Gender: {result.get('dominant_gender', '')} | "
                f"Race: {result.get('dominant_race', '')} | "
                f"Age: {result.get('age', '')} (NOT SO Accurate)"
            )
            cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        else:
            cv2.putText(frame, "Analyzing... or no face detected.", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Emotion Recognition - Press 'q' to Exit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[-] Interrupted by user.")

# Cleanup
running = False
thread.join()
cap.release()
cv2.destroyAllWindows()

print("[-]Exited at:", datetime.now().strftime("%I:%M %p"), "On", datetime.now().strftime("%d %B %Y, %A"))
print("------------------------------------------------------- :COMPLETE: --------------------------------------------------------------------------------------")
