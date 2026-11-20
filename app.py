"""
SignBridge - Indian Sign Language (ISL) Real-Time Translator.
Modern desktop dashboard featuring live landmark overlay, temporal smoothing,
sentence accumulator, audio playback, and confidence analytics.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import time
import subprocess
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np

# Ensure root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.core.landmarks import LandmarkExtractor
from src.core.classifier import SignLanguageClassifier
from src.core.tts_engine import TTSEngine
from src.core.sentence_builder import SentenceBuilder
from src.utils.config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    MODEL_PATH,
    LABELS_PATH,
    NORMALIZE_LANDMARKS,
    PREDICTION_CONFIDENCE_THRESHOLD,
    TTS_BACKEND,
    TTS_RATE,
    TTS_VOLUME
)

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ModernSignLanguageApp:
    """Integrated Desktop GUI for Sign Language Translation."""

    def __init__(self, root):
        self.root = root
        self.root.title("SignBridge - Indian Sign Language Translator")
        self.root.geometry("1100x720")
        self.root.minsize(960, 640)
        self.root.configure(bg="#121214")

        # Core Engines
        self.extractor = LandmarkExtractor(max_num_hands=1)
        self.classifier = SignLanguageClassifier(model_path=MODEL_PATH, labels_path=LABELS_PATH)
        self.tts = TTSEngine(backend=TTS_BACKEND, rate=TTS_RATE, volume=TTS_VOLUME)
        self.sentence_builder = SentenceBuilder(confidence_threshold=PREDICTION_CONFIDENCE_THRESHOLD)

        # State Variables
        self.cap = None
        self.is_camera_running = False
        self.show_skeleton = tk.BooleanVar(value=True)
        self.auto_speak_words = tk.BooleanVar(value=False)
        self.prev_frame_time = time.time()
        self.fps = 0.0
        self.canvas_image_id = None

        # UI Variables
        self.current_sign_var = tk.StringVar(value="--")
        self.confidence_var = tk.StringVar(value="Confidence: 0%")
        self.sentence_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready. Click 'Start Camera' to begin.")

        self._apply_dark_theme()
        self._build_layout()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _apply_dark_theme(self):
        """Configure ttk styles for dark modern aesthetic."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure(".", background="#121214", foreground="#E4E4E7")
        self.style.configure("TFrame", background="#18181B")
        self.style.configure("Card.TFrame", background="#222226", relief="flat")

        self.style.configure("Header.TLabel", background="#121214", foreground="#F4F4F5",
                             font=("Segoe UI", 18, "bold"))
        self.style.configure("SubHeader.TLabel", background="#121214", foreground="#A1A1AA",
                             font=("Segoe UI", 9))

        self.style.configure("CardTitle.TLabel", background="#222226", foreground="#93C5FD",
                             font=("Segoe UI", 10, "bold"))
        self.style.configure("SignDisplay.TLabel", background="#222226", foreground="#34D399",
                             font=("Segoe UI", 26, "bold"))
        self.style.configure("Meta.TLabel", background="#222226", foreground="#A1A1AA",
                             font=("Segoe UI", 9))

        self.style.configure("Primary.TButton", background="#2563EB", foreground="#FFFFFF",
                             font=("Segoe UI", 10, "bold"), padding=6)
        self.style.map("Primary.TButton", background=[("active", "#1D4ED8")])

        self.style.configure("Secondary.TButton", background="#3F3F46", foreground="#F4F4F5",
                             font=("Segoe UI", 9), padding=5)
        self.style.map("Secondary.TButton", background=[("active", "#52525B")])

        self.style.configure("Accent.TButton", background="#059669", foreground="#FFFFFF",
                             font=("Segoe UI", 10, "bold"), padding=6)
        self.style.map("Accent.TButton", background=[("active", "#047857")])

        self.style.configure("TProgressbar", thickness=8, troughcolor="#27272A", background="#3B82F6")

    def _build_layout(self):
        """Construct responsive split-pane UI."""
        # Top Header Bar
        header = tk.Frame(self.root, bg="#121214", padx=16, pady=12)
        header.pack(fill=tk.X)

        title_lbl = ttk.Label(header, text="SignBridge", style="Header.TLabel")
        title_lbl.pack(side=tk.LEFT)
        sub_lbl = ttk.Label(header, text="  |  Indian Sign Language Real-Time Voice Translation",
                            style="SubHeader.TLabel")
        sub_lbl.pack(side=tk.LEFT, pady=(5, 0))

        self.fps_lbl = ttk.Label(header, text="FPS: 0.0", style="SubHeader.TLabel")
        self.fps_lbl.pack(side=tk.RIGHT, pady=(5, 0))

        # Main Workspace Container
        content = tk.Frame(self.root, bg="#121214", padx=16, pady=8)
        content.pack(fill=tk.BOTH, expand=True)

        # Left Column: Video Feed & Controls
        left_pane = tk.Frame(content, bg="#18181B", bd=1, relief="solid")
        left_pane.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video Canvas
        self.canvas_width = 640
        self.canvas_height = 480
        self.video_canvas = tk.Canvas(left_pane, width=self.canvas_width, height=self.canvas_height,
                                      bg="#09090B", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Video Action Controls
        control_bar = tk.Frame(left_pane, bg="#18181B", padx=8, pady=8)
        control_bar.pack(fill=tk.X)

        self.toggle_cam_btn = ttk.Button(control_bar, text="▶ Start Camera", style="Primary.TButton",
                                         command=self.toggle_camera)
        self.toggle_cam_btn.pack(side=tk.LEFT, padx=(0, 8))

        skeleton_chk = tk.Checkbutton(control_bar, text="Hand Skeleton Overlay", variable=self.show_skeleton,
                                      bg="#18181B", fg="#E4E4E7", selectcolor="#27272A",
                                      activebackground="#18181B", activeforeground="#FFFFFF",
                                      font=("Segoe UI", 9))
        skeleton_chk.pack(side=tk.LEFT, padx=8)

        # Right Column: Inference Analytics, Sentence Builder, Settings
        right_pane = tk.Frame(content, bg="#121214", width=360)
        right_pane.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(0, 0))
        right_pane.pack_propagate(False)

        # Card 1: Real-Time Prediction Card
        card1 = ttk.Frame(right_pane, style="Card.TFrame", padding=12)
        card1.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(card1, text="ACTIVE GESTURE", style="CardTitle.TLabel").pack(anchor=tk.W)
        self.sign_lbl = ttk.Label(card1, textvariable=self.current_sign_var, style="SignDisplay.TLabel")
        self.sign_lbl.pack(anchor=tk.CENTER, pady=8)

        self.conf_lbl = ttk.Label(card1, textvariable=self.confidence_var, style="Meta.TLabel")
        self.conf_lbl.pack(anchor=tk.W)

        self.conf_bar = ttk.Progressbar(card1, style="TProgressbar", maximum=100)
        self.conf_bar.pack(fill=tk.X, pady=(4, 8))

        ttk.Label(card1, text="Hold Duration Commitment:", style="Meta.TLabel").pack(anchor=tk.W)
        self.hold_bar = ttk.Progressbar(card1, style="TProgressbar", maximum=100)
        self.hold_bar.pack(fill=tk.X, pady=(4, 0))

        # Card 2: Sentence Construction Hub
        card2 = ttk.Frame(right_pane, style="Card.TFrame", padding=12)
        card2.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(card2, text="SENTENCE ACCUMULATOR", style="CardTitle.TLabel").pack(anchor=tk.W)

        self.sentence_box = tk.Text(card2, bg="#18181B", fg="#F4F4F5", insertbackground="#F4F4F5",
                                    font=("Segoe UI", 12), height=4, relief="flat", wrap=tk.WORD,
                                    padx=8, pady=8)
        self.sentence_box.pack(fill=tk.BOTH, expand=True, pady=8)
        self.sentence_box.insert("1.0", "Signed words accumulate here into full sentences...")
        self.sentence_box.config(state=tk.DISABLED)

        # Sentence Actions
        btn_grid = tk.Frame(card2, bg="#222226")
        btn_grid.pack(fill=tk.X)

        speak_btn = ttk.Button(btn_grid, text="🔊 Speak Sentence", style="Accent.TButton",
                               command=self.speak_full_sentence)
        speak_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        back_btn = ttk.Button(btn_grid, text="⌫ Undo", style="Secondary.TButton",
                              command=self.undo_last_sign)
        back_btn.pack(side=tk.LEFT, padx=4)

        clear_btn = ttk.Button(btn_grid, text="🗑 Clear", style="Secondary.TButton",
                               command=self.clear_sentence)
        clear_btn.pack(side=tk.LEFT, padx=(4, 0))

        # Card 3: Audio & System Settings
        card3 = ttk.Frame(right_pane, style="Card.TFrame", padding=12)
        card3.pack(fill=tk.X)

        ttk.Label(card3, text="AUDIO & OUTPUT SETTINGS", style="CardTitle.TLabel").pack(anchor=tk.W)

        auto_chk = tk.Checkbutton(card3, text="Auto-speak each word immediately", variable=self.auto_speak_words,
                                  bg="#222226", fg="#E4E4E7", selectcolor="#27272A",
                                  activebackground="#222226", activeforeground="#FFFFFF",
                                  font=("Segoe UI", 9))
        auto_chk.pack(anchor=tk.W, pady=(4, 8))

        train_btn = ttk.Button(card3, text="🛠 Open Dataset Studio & Trainer", style="Secondary.TButton",
                               command=self.launch_trainer_tool)
        train_btn.pack(fill=tk.X)

        # Bottom Status Bar
        status_bar = tk.Frame(self.root, bg="#18181B", padx=16, pady=4)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_lbl = ttk.Label(status_bar, textvariable=self.status_var, style="SubHeader.TLabel")
        self.status_lbl.pack(side=tk.LEFT)

    def toggle_camera(self):
        """Start or stop the live camera stream."""
        if not self.is_camera_running:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", f"Could not access camera at index {CAMERA_INDEX}.")
                return

            self.is_camera_running = True
            self.toggle_cam_btn.configure(text="⏹ Stop Camera", style="Secondary.TButton")
            self.status_var.set("Camera active. Tracking hands...")
            self._update_video_stream()
        else:
            self.is_camera_running = False
            if self.cap:
                self.cap.release()
            self.toggle_cam_btn.configure(text="▶ Start Camera", style="Primary.TButton")
            self.status_var.set("Camera stopped.")
            self.current_sign_var.set("--")
            self.confidence_var.set("Confidence: 0%")
            self.conf_bar["value"] = 0
            self.hold_bar["value"] = 0

    def _update_video_stream(self):
        """Frame processing loop triggered via Tkinter's after timer."""
        if not self.is_camera_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(20, self._update_video_stream)
            return

        frame = cv2.flip(frame, 1)  # Mirror view
        results = self.extractor.process(frame)

        # Calculate FPS
        now = time.time()
        dt = now - self.prev_frame_time
        self.prev_frame_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
            self.fps_lbl.configure(text=f"FPS: {self.fps:.1f}")

        # Extract features & predict
        features = self.extractor.extract_features(results, mode="single", normalize=NORMALIZE_LANDMARKS)

        raw_label = None
        confidence = 0.0

        if features is not None and self.classifier.model is not None:
            try:
                _, raw_label, confidence, _ = self.classifier.predict(features)
            except Exception as e:
                print(f"[Prediction Error] {e}")

        # Update Sentence Builder & Temporal Smoothing
        smoothed_label, hold_progress, committed_word = self.sentence_builder.update(raw_label, confidence)

        # Update UI state
        if smoothed_label:
            self.current_sign_var.set(smoothed_label.upper())
            self.confidence_var.set(f"Confidence: {confidence * 100:.1f}%")
            self.conf_bar["value"] = int(confidence * 100)
            self.hold_bar["value"] = int(hold_progress * 100)
        else:
            self.current_sign_var.set("--")
            self.confidence_var.set("Confidence: 0%")
            self.conf_bar["value"] = 0
            self.hold_bar["value"] = 0

        # Handle committed word emission
        if committed_word:
            self._sync_sentence_display()
            if self.auto_speak_words.get():
                self.tts.speak(committed_word)

        # Draw overlays on frame
        if self.show_skeleton.get():
            self.extractor.draw_landmarks(frame, results)

        # Convert frame to PhotoImage and render to canvas (reuse single item to avoid memory leak)
        if PIL_AVAILABLE:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            target_w = self.video_canvas.winfo_width()
            target_h = self.video_canvas.winfo_height()
            target_w = target_w if target_w > 10 else self.canvas_width
            target_h = target_h if target_h > 10 else self.canvas_height
            img = img.resize((target_w, target_h))
            self.photo_img = ImageTk.PhotoImage(image=img)
            if self.canvas_image_id is None:
                self.canvas_image_id = self.video_canvas.create_image(0, 0, image=self.photo_img, anchor=tk.NW)
            else:
                self.video_canvas.itemconfig(self.canvas_image_id, image=self.photo_img)

        # Loop
        self.root.after(15, self._update_video_stream)

    def _sync_sentence_display(self):
        """Update sentence text box with currently formed sentence."""
        sentence = self.sentence_builder.get_sentence()
        self.sentence_box.config(state=tk.NORMAL)
        self.sentence_box.delete("1.0", tk.END)
        self.sentence_box.insert("1.0", sentence if sentence else "Signed words accumulate here...")
        self.sentence_box.config(state=tk.DISABLED)

    def speak_full_sentence(self):
        """Speak the entire accumulated sentence."""
        sentence = self.sentence_builder.get_sentence()
        if sentence:
            self.status_var.set(f"Speaking: '{sentence}'")
            self.tts.speak(sentence)
        else:
            self.status_var.set("Sentence buffer is empty.")

    def undo_last_sign(self):
        """Remove last signed word."""
        removed = self.sentence_builder.backspace()
        self._sync_sentence_display()
        if removed:
            self.status_var.set(f"Removed word: '{removed}'")

    def clear_sentence(self):
        """Clear sentence buffer."""
        self.sentence_builder.clear()
        self._sync_sentence_display()
        self.status_var.set("Sentence cleared.")

    def launch_trainer_tool(self):
        """Release camera if active and launch the interactive Dataset Studio."""
        if self.is_camera_running:
            self.toggle_camera()
        try:
            subprocess.Popen([sys.executable, str(BASE_DIR / "train.py")])
            self.status_var.set("Launched Dataset Studio (train.py).")
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch trainer:\n{e}")

    def on_close(self):
        """Clean shutdown of camera and threads."""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
        self.extractor.close()
        self.tts.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = ModernSignLanguageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
