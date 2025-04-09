from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import json
import threading
import sounddevice as sd
from speechbrain.inference import Tacotron2, HIFIGAN

app = Flask(__name__)

# Load model and gesture labels
model = tf.keras.models.load_model("sign_language_model.h5")
with open("gesture_labels.json") as f:
    gesture_mapping = json.load(f)

# Load TTS
print("Loading TTS Models...")
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tts_model", run_opts={"device": "cpu"})
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="vocoder_model", run_opts={"device": "cpu"})
print("TTS Models Loaded!")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Global vars
cap = cv2.VideoCapture(0)
last_spoken = None
last_spoken_time = 0
speak_delay = 1.5

def speak(text):
    def _play():
        mel_output, _, _ = tacotron2.encode_text(text)
        waveforms = hifi_gan.decode_batch(mel_output)
        audio = waveforms.squeeze().cpu().numpy()
        sd.play(audio, samplerate=22050)
        sd.wait()
    threading.Thread(target=_play, daemon=True).start()

def generate_frames():
    global last_spoken, last_spoken_time
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)

                prediction = model.predict(landmarks)
                predicted_class = np.argmax(prediction)
                gesture = gesture_mapping.get(str(predicted_class), "Unknown")

                if gesture != last_spoken and (time.time() - last_spoken_time) > speak_delay:
                    speak(gesture)
                    last_spoken = gesture
                    last_spoken_time = time.time()

                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
