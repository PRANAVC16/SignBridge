"""
Multi-backend Text-To-Speech (TTS) Engine.
Provides low-latency offline voice synthesis (pyttsx3) alongside high-fidelity neural models (SpeechBrain).
Uses a thread-safe decoupled task queue to prevent UI or video stream freezing.
"""
import os
import queue
import threading
import time


class TTSEngine:
    """Thread-safe multi-backend text-to-speech engine."""

    def __init__(self, backend="pyttsx3", rate=160, volume=1.0):
        """
        Initialize the TTS engine.
        
        Args:
            backend: "pyttsx3" (fast, offline), "speechbrain" (neural), or "console" (silent fallback).
            rate: Speech rate in words per minute.
            volume: Volume level from 0.0 to 1.0.
        """
        self.backend_name = backend
        self.rate = rate
        self.volume = volume
        self.is_running = True
        self.speech_queue = queue.Queue(maxsize=10)

        # Backend specific attributes
        self.pyttsx3_engine = None
        self.sb_tacotron = None
        self.sb_hifigan = None

        # Start background worker thread (initializes backend on worker thread for Windows COM safety)
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _init_backend(self):
        """Initialize the selected speech synthesis backend on the worker thread."""
        if self.backend_name == "pyttsx3":
            try:
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except ImportError:
                    pass
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                engine.setProperty('volume', self.volume)
                self.pyttsx3_engine = engine
            except Exception as e:
                print(f"[TTS Warning] pyttsx3 initialization failed ({e}). Falling back to console mode.")
                self.backend_name = "console"

        elif self.backend_name == "speechbrain":
            try:
                import sounddevice as sd
                from speechbrain.inference import Tacotron2, HIFIGAN
                print("[TTS] Loading SpeechBrain Tacotron2 & HiFiGAN models...")
                self.sb_tacotron = Tacotron2.from_hparams(
                    source="speechbrain/tts-tacotron2-ljspeech",
                    savedir="tts_model",
                    run_opts={"device": "cpu"}
                )
                self.sb_hifigan = HIFIGAN.from_hparams(
                    source="speechbrain/tts-hifigan-ljspeech",
                    savedir="vocoder_model",
                    run_opts={"device": "cpu"}
                )
                self.sd = sd
                print("[TTS] SpeechBrain neural models loaded successfully.")
            except Exception as e:
                print(f"[TTS Warning] SpeechBrain failed to load ({e}). Falling back to console mode.")
                self.backend_name = "console"

        elif self.backend_name == "console":
            pass
        else:
            print(f"[TTS Warning] Unknown backend '{self.backend_name}'. Defaulting to console mode.")
            self.backend_name = "console"

    def _speech_worker(self):
        """Worker loop executing speech jobs from the queue."""
        self._init_backend()
        while self.is_running:
            try:
                text = self.speech_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._synthesize_and_play(text)
            except Exception as e:
                print(f"[TTS Error] Failed to synthesize '{text}': {e}")
            finally:
                self.speech_queue.task_done()

    def _synthesize_and_play(self, text):
        """Synthesize and play audio based on active backend."""
        if not text or not text.strip():
            return

        clean_text = text.strip()

        if self.backend_name == "pyttsx3" and self.pyttsx3_engine:
            self.pyttsx3_engine.say(clean_text)
            self.pyttsx3_engine.runAndWait()

        elif self.backend_name == "speechbrain" and self.sb_tacotron and self.sb_hifigan:
            print(f"[TTS SpeechBrain] Synthesizing: {clean_text}")
            mel_output, _, _ = self.sb_tacotron.encode_text(clean_text)
            waveforms = self.sb_hifigan.decode_batch(mel_output)
            audio = waveforms.squeeze().cpu().numpy()
            self.sd.play(audio, samplerate=22050)
            self.sd.wait()

        else:
            print(f"[TTS Voice Output]: \"{clean_text}\"")

    def speak(self, text, flush_previous=False):
        """
        Enqueue text for spoken synthesis.
        
        Args:
            text: String to speak.
            flush_previous: If True, clears queued pending utterances first.
        """
        if not text:
            return

        if flush_previous:
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                except (queue.Empty, ValueError):
                    break

        try:
            self.speech_queue.put_nowait(text)
        except queue.Full:
            pass  # Drop if queue is saturated

    def set_rate(self, rate):
        """Set speech rate (words per minute)."""
        self.rate = rate
        if self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('rate', rate)

    def set_volume(self, volume):
        """Set volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))
        if self.pyttsx3_engine:
            self.pyttsx3_engine.setProperty('volume', self.volume)

    def stop(self):
        """Stop worker and clean up."""
        self.is_running = False
        if hasattr(self, 'pyttsx3_engine') and self.pyttsx3_engine:
            try:
                self.pyttsx3_engine.stop()
            except Exception:
                pass
