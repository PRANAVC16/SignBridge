"""
Interactive Dataset Collection & Neural Network Training Tool for ISL.
Captures spatial hand landmarks in real-time, saves labeled datasets,
and trains an invariant Keras classifier with validation reporting.
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.landmarks import LandmarkExtractor
from src.core.classifier import SignLanguageClassifier
from src.utils.config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MODEL_PATH,
    LABELS_PATH,
    DATA_DIR,
    NORMALIZE_LANDMARKS
)

DATASET_FILE = DATA_DIR / "gesture_dataset.npz"


def run_trainer():
    """Main interactive collection and training loop."""
    print("=" * 60)
    print(" Indian Sign Language - Dataset Collector & Trainer")
    print("=" * 60)

    # Initialize components
    extractor = LandmarkExtractor(max_num_hands=1)
    classifier = SignLanguageClassifier(labels_path=LABELS_PATH)

    labels = classifier.labels
    samples_data = []
    samples_labels = []

    # Load existing dataset if available
    if os.path.exists(DATASET_FILE):
        try:
            data = np.load(DATASET_FILE, allow_pickle=True)
            samples_data = list(data["X"])
            samples_labels = list(data["y"])
            print(f" Loaded {len(samples_data)} existing samples from {DATASET_FILE.name}")
        except Exception as e:
            print(f" Could not load existing dataset: {e}")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"Error: Unable to open camera {CAMERA_INDEX}")
        return

    recording_gesture_idx = None
    is_recording = False
    status_msg = "Press [1-9] to select/record gesture. [T] to Train. [ESC] to Quit."

    print("\nControls:")
    print("  [1-9] : Toggle continuous recording for gesture slot")
    print("  [SPACE]: Record a single sample for active slot")
    print("  [N]   : Name/rename the current gesture slot")
    print("  [T]   : Train Neural Network on collected dataset")
    print("  [S]   : Save dataset to disk")
    print("  [ESC] : Exit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera feed lost.")
            break

        frame = cv2.flip(frame, 1)  # Mirror feed
        results = extractor.process(frame)
        features = extractor.extract_features(results, mode="single", normalize=NORMALIZE_LANDMARKS)

        # Draw skeleton
        extractor.draw_landmarks(frame, results)

        # Record if active and hand detected
        if is_recording and recording_gesture_idx is not None and features is not None:
            samples_data.append(features)
            samples_labels.append(recording_gesture_idx)

        # Render On-Screen Display (HUD)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 85), (20, 20, 20), -1)

        slot_display = str(recording_gesture_idx + 1) if recording_gesture_idx is not None else "-"
        slot_name = labels.get(str(recording_gesture_idx), "Unassigned") if recording_gesture_idx is not None else "None"
        rec_status = "RECORDING" if is_recording else "IDLE"
        rec_color = (0, 0, 255) if is_recording else (180, 180, 180)

        cv2.putText(frame, f"Slot: {slot_display} ({slot_name}) | Status: {rec_status}",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, rec_color, 2)

        # Count samples for current gesture
        curr_count = samples_labels.count(recording_gesture_idx) if recording_gesture_idx is not None else 0
        total_count = len(samples_labels)
        cv2.putText(frame, f"Slot Samples: {curr_count} | Total Samples: {total_count}",
                    (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.putText(frame, status_msg, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow("Sign Language Studio - Dataset Trainer", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break

        # Select Slot [1-9]
        elif ord('1') <= key <= ord('9'):
            slot_idx = int(chr(key)) - 1
            if recording_gesture_idx == slot_idx:
                is_recording = not is_recording
            else:
                recording_gesture_idx = slot_idx
                is_recording = True
                if str(slot_idx) not in labels:
                    labels[str(slot_idx)] = f"Gesture_{slot_idx + 1}"

        # Single capture [SPACE]
        elif key == ord(' '):
            if recording_gesture_idx is not None and features is not None:
                samples_data.append(features)
                samples_labels.append(recording_gesture_idx)
                status_msg = f"Captured single sample for {labels.get(str(recording_gesture_idx))}"

        # Rename gesture [N]
        elif key in (ord('n'), ord('N')):
            if recording_gesture_idx is not None:
                cv2.putText(frame, "ENTER NEW NAME IN TERMINAL...", (w // 4, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Sign Language Studio - Dataset Trainer", frame)
                cv2.waitKey(10)
                new_name = input(f"\nEnter name for gesture slot {recording_gesture_idx + 1}: ").strip()
                if new_name:
                    labels[str(recording_gesture_idx)] = new_name
                    classifier.labels = labels
                    classifier.save_labels(LABELS_PATH)
                    status_msg = f"Renamed slot {recording_gesture_idx} to '{new_name}'"

        # Save Dataset [S]
        elif key in (ord('s'), ord('S')):
            if samples_data:
                np.savez_compressed(DATASET_FILE, X=np.array(samples_data), y=np.array(samples_labels))
                classifier.save_labels(LABELS_PATH)
                status_msg = f"Saved {len(samples_data)} samples to {DATASET_FILE.name}"
                print(f"[Dataset] Saved to {DATASET_FILE}")

        # Train Model [T]
        elif key in (ord('t'), ord('T')):
            if len(set(samples_labels)) < 2:
                status_msg = "Need at least 2 distinct gesture classes to train!"
                print(status_msg)
            else:
                is_recording = False
                cv2.putText(frame, "TRAINING NEURAL NETWORK... CHECK TERMINAL", (20, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Sign Language Studio - Dataset Trainer", frame)
                cv2.waitKey(50)

                print("\n[Training] Starting Neural Network Training...")
                X_arr = np.array(samples_data, dtype=np.float32)

                # Remap slot IDs to contiguous class indices [0, ..., K-1]
                unique_slots = sorted(set(samples_labels))
                slot_to_class = {slot: idx for idx, slot in enumerate(unique_slots)}
                y_arr = np.array([slot_to_class[s] for s in samples_labels], dtype=np.int32)
                remapped_labels = {
                    str(idx): labels.get(str(slot), f"Gesture_{slot + 1}")
                    for slot, idx in slot_to_class.items()
                }
                classifier.labels = remapped_labels

                # Save dataset and remapped labels
                np.savez_compressed(DATASET_FILE, X=X_arr, y=y_arr)
                classifier.save_labels(LABELS_PATH)

                # Train
                history = classifier.train(X_arr, y_arr, epochs=35, batch_size=16, val_split=0.2)
                classifier.save_model(MODEL_PATH)
                val_acc = history.history.get('val_accuracy', history.history.get('accuracy', [0.0]))[-1]
                status_msg = f"Model trained & saved! Final Acc: {val_acc:.2%}"
                print(f"\n[Success] {status_msg}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("Trainer closed.")


if __name__ == "__main__":
    run_trainer()
