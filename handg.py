import os
os.environ["HUGGINGFACE_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading
import time
import json
import sounddevice as sd
import speechbrain as sb


# Import TTS models from the correct place
from speechbrain.inference import Tacotron2, HIFIGAN

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Camera
cap = cv2.VideoCapture(0)

# Load TTS Models
print("Loading TTS Models...")
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tts_model", run_opts={"device": "cpu"})
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="vocoder_model", run_opts={"device": "cpu"})

print("TTS Models Loaded!")


# File names
model_filename = "sign_language_model.h5"
gesture_labels_file = "gesture_labels.json"

# Load Gesture Labels
gesture_mapping = {}

def load_labels():
    global gesture_mapping
    if os.path.exists(gesture_labels_file):
        with open(gesture_labels_file, "r") as f:
            gesture_mapping = json.load(f)
        print("Gesture labels loaded:", gesture_mapping)
    else:
        print("No saved gesture labels found.")

def save_labels():
    with open(gesture_labels_file, "w") as f:
        json.dump(gesture_mapping, f)
    print("Gesture labels saved!")

# Load Model if Exists
if os.path.exists(model_filename):
    model = tf.keras.models.load_model(model_filename)
    load_labels()
else:
    model = None

# -------------------- TRAINING MODE --------------------
def train_mode():
    global gesture_mapping
    training_data = []
    training_labels = []
    collected_gestures = {}

    print("Training Mode: Press a number key (1-9) to start capturing that gesture. Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Camera not found.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                key = cv2.waitKey(1) & 0xFF
                if key in range(ord('1'), ord('9')+1):  # Press 1-9 to capture gesture
                    gesture_index = int(chr(key)) - 1
                    
                    # Check if this gesture has already been named
                    if gesture_index not in collected_gestures:
                        label_name = input(f"Enter name for gesture {gesture_index + 1}: ")
                        gesture_mapping[gesture_index] = label_name
                        collected_gestures[gesture_index] = label_name
                        print(f"Gesture '{label_name}' assigned to key {gesture_index + 1}")

                    training_data.append(landmarks)
                    training_labels.append(gesture_index)
                    print(f"Captured {collected_gestures[gesture_index]} gesture sample.")

        cv2.imshow("Training Mode", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC pressed
            print("Training Mode Exited.")
            break

    cv2.destroyAllWindows()

    if training_data:
        save_labels()
        train_model(np.array(training_data), np.array(training_labels))

# -------------------- MODEL TRAINING --------------------
def train_model(X, y):
    global model

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(set(y)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X, y, epochs=30, batch_size=16, verbose=1)
    model.save(model_filename)
    print("Model Trained and Saved!")

# -------------------- RECOGNITION MODE --------------------
def recognition_mode():
    global model
    if os.path.exists(model_filename):
        model = tf.keras.models.load_model(model_filename)
        load_labels()
    else:
        model = None
    if model is None:
        print("Error: Model not trained. Please train first.")
        return

    print("Recognition Mode: Press ESC to stop.")
    last_spoken = None
    last_spoken_time = 0
    speak_delay = 1.5  # Delay to avoid repeated speech

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)

                prediction = model.predict(landmarks)
                predicted_class = np.argmax(prediction)

                # Print model output for debugging
                print(f"Raw Model Output: {prediction}")
                print(f"Predicted Class Index: {predicted_class}")

                recognized_gesture = gesture_mapping.get(str(predicted_class), "Unknown")
                print(f"Recognized Gesture: {recognized_gesture}")

                cv2.putText(frame, f"Recognized: {recognized_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if recognized_gesture != last_spoken and (time.time() - last_spoken_time) > speak_delay:
                    speak(recognized_gesture)
                    last_spoken = recognized_gesture
                    last_spoken_time = time.time()

        else:
            cv2.putText(frame, "No hand detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC pressed
            print("Exiting Recognition Mode.")
            break

    cv2.destroyAllWindows()

# -------------------- SPEECH FUNCTION --------------------
import threading

def speak(text):
    """Convert text to speech using SpeechBrain TTS and play asynchronously."""
    def _play_audio():
        print(f"Speaking: {text}")
        mel_output, _, _ = tacotron2.encode_text(text)
        waveforms = hifi_gan.decode_batch(mel_output)
        audio = waveforms.squeeze().cpu().numpy()
        sd.play(audio, samplerate=22050)
        sd.wait()

    # Run speech in a separate thread
    speech_thread = threading.Thread(target=_play_audio, daemon=True)
    speech_thread.start()

# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    while True:
        print("\nOptions:")
        print("1. Train Gestures")
        print("2. Recognize Gestures")
        print("3. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            train_mode()
        elif choice == "2":
            recognition_mode()
        elif choice == "3":
            break
        else:
            print("Invalid choice! Try again.")

    cap.release()
    cv2.destroyAllWindows()
    print("Program exited successfully.")
