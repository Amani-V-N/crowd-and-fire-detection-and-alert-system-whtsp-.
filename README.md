# crowd-and-fire-detection-and-alert-system-whtsp-.
import cv2
import time
import base64
import requests
import threading
import numpy as np
import winsound
import pyttsx3
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from collections import deque
from twilio.rest import Client


# ==============================
# YOLO MODEL
# ==============================
yolo = YOLO("yolov8n.pt")

# ==============================
# WHATSAPP (TWILIO) CONFIG
# ==============================

ACCOUNT_SID = "AC65df15166a3ec4fadb68d14ccd03bbb0"
AUTH_TOKEN = "310f9cde5b34ebbc44189e9a7833f127"

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_whatsapp_image(message, image_url):
    client.messages.create(
        from_="whatsapp:+14155238886",   # Twilio sandbox number
        to="whatsapp:+919946046480",     # YOUR phone number
        body=message,
        media_url=[image_url]
    )
image_url = "https://neonatally-fenestral-carlee.ngrok-free.dev/image/fire.jpg"

send_whatsapp_image(
    "🔥 Fire detected! Immediate action required.",
    image_url
)


twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)


# ==============================
# SYSTEM CONFIG
# ==============================
CROWD_THRESHOLD = 6
ALERT_GAP = 6
LOG_FILE = "alert_log.csv"

# ==============================
# TTS (INIT ONCE)
# ==============================
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)

def voice_alert(message):
    engine.say(message)
    engine.runAndWait()


# ==============================
# LOGGING
# ==============================
def log_alert(alert_type, message):
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Time", "Type", "Message"])
        writer.writerow([datetime.now(), alert_type, message])

# ==============================
# whtsp
# ==============================
# ==============================
# WHATSAPP TEXT ALERT
# ==============================
def send_whatsapp_alert(message):
    try:
        from twilio.rest import Client

        ACCOUNT_SID = "AC65df15166a3ec4fadb68d14ccd03bbb0"   # your SID
        AUTH_TOKEN = "310f9cde5b34ebbc44189e9a7833f127"      # your token

        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        client.messages.create(
            from_="whatsapp:+14155238886",   # Twilio sandbox
            to="whatsapp:+919946046480",     # YOUR phone
            body=message
        )

    except Exception as e:
        print("WhatsApp text error:", e)


# ==============================
# ALERT SYSTEM
# ==============================
last_alert_time = 0

def play_alarm(alert_type):
    if alert_type == "fire":
        # 🔥 VERY LOUD + REPEATED fire alarm
        for _ in range(5):
            winsound.Beep(2500, 700)  # high pitch
            time.sleep(0.1)
    else:
        # normal alert
        winsound.Beep(1500, 400)


def voice_alert(msg):
    engine.say(msg)
    engine.runAndWait()

def send_whatsapp_image(frame, caption):
    try:
        # Save image locally
        os.makedirs("alerts", exist_ok=True)
        filename = f"alert_{int(time.time())}.jpg"
        filepath = os.path.join("alerts", filename)
        cv2.imwrite(filepath, frame)

        # ⚠️ TEMPORARY FIX FOR DEMO:
        # Send ONLY TEXT if image URL not configured
        # (Image hosting like ngrok needed for real images)

        

    except Exception as e:
        print("WhatsApp image error:", e)
# ==============================
# WHATSAPP / PHONE VOICE CALL
# ==============================
def make_voice_call(message):
    try:
        from twilio.rest import Client

        ACCOUNT_SID = "AC65df15166a3ec4fadb68d14ccd03bbb0"
        AUTH_TOKEN = "310f9cde5b34ebbc44189e9a7833f127"

        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        call = client.calls.create(
            to="+919946046480",          # YOUR phone number
            from_="+14155238886",        # Twilio voice number
            twiml=f'<Response><Say voice="alice">{message}</Say></Response>'
        )

        print("Voice call sent")

    except Exception as e:
        print("Voice call error:", e)


def trigger_alert(alert_type, message, frame):
    global last_alert_time
    if time.time() - last_alert_time < ALERT_GAP:
        return

    log_alert(alert_type, message)

    # 🔊 SOUND (LOUD FOR FIRE)
    threading.Thread(
        target=play_alarm,
        args=(alert_type,),
        daemon=True
    ).start()

    # 🗣️ VOICE
    threading.Thread(
        target=voice_alert,
        args=(message,),
        daemon=True
    ).start()

    # 📩 WhatsApp text
    threading.Thread(
        target=send_whatsapp_alert,
        args=(f"⚠️ {message}",),
        daemon=True
    ).start()

    # 📸 WhatsApp image
    threading.Thread(
        target=send_whatsapp_image,
        args=(frame.copy(), message),
        daemon=True
    ).start()
#whatsapp voice call
    last_alert_time = time.time()
    threading.Thread(
    target=make_voice_call,
    args=(message,),
    daemon=True
).start()





    

# ==============================
# MOONDREAM (OLLAMA VLM)
# ==============================
def moondream_reason(frame):
    _, buf = cv2.imencode(".jpg", frame)
    img_b64 = base64.b64encode(buf).decode()

    payload = {
        "model": "moondream",
        "prompt": (
            "Is there any fire, flames, burning logs, or fireplace visible? "
            "Answer only yes or no."
        ),
        "images": [img_b64],
        "stream": False
    }

    try:
        r = requests.post("http://localhost:11434/api/generate",
                          json=payload, timeout=25)
        return r.json().get("response", "").lower()
    except:
        return ""

# ==============================
# FIRE DETECTION (COLOR + FLICKER)
# ==============================
prev_gray = None

def detect_fire_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([5, 170, 170])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return cv2.countNonZero(mask) > 5000

def detect_fire_motion(frame):
    global prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        return False

    diff = cv2.absdiff(prev_gray, gray)
    prev_gray = gray
    return np.mean(diff) > 2.0   # flicker check

# ==============================
# CAMERA
# ==============================
cap = cv2.VideoCapture(0)

scene_text = ""
last_vlm_time = 0
danger_history = deque(maxlen=5)

def run_vlm(frame):
    global scene_text
    scene_text = moondream_reason(frame)
    print("VLM:", scene_text)

# ==============================
# MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    danger = "LOW"
    status = "SAFE"

    small = cv2.resize(frame, (640, 480))

    # PEOPLE COUNT
    results = yolo(small, conf=0.4, verbose=False)[0]
    people_count = 0
    for box in results.boxes:
        if int(box.cls[0]) == 0:
            people_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # FIRE (DEMO SAFE)
    fire_color = detect_fire_color(small)
    fire_motion = detect_fire_motion(small)
    fire_vlm = any(w in scene_text for w in ["yes","fire","flame","burning","fireplace"])

    if (fire_color and fire_motion) or fire_vlm:
        danger = "HIGH"
        status = "🔥 FIRE ALERT"
        trigger_alert("fire","🔥 Fire detected! Immediate action required.", frame)

    elif people_count >= CROWD_THRESHOLD:
        danger = "MEDIUM"
        status = "OVERCROWDED"
        trigger_alert("crowd","Cafe overcrowded", frame)

    if time.time() - last_vlm_time > 2:
        last_vlm_time = time.time()
        threading.Thread(target=run_vlm, args=(frame.copy(),), daemon=True).start()

    danger_history.append(danger)
    danger = max(set(danger_history), key=danger_history.count)

    color = (0,255,0) if danger=="LOW" else (0,255,255) if danger=="MEDIUM" else (0,0,255)

    cv2.putText(frame, f"DANGER: {danger}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    cv2.putText(frame, f"STATUS: {status}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"People: {people_count}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Cafe Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
