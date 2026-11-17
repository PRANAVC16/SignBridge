# SignBridge: Portfolio & Interview Guide

This guide is designed to help you showcase **SignBridge** on your resume, LinkedIn, GitHub, and during technical interviews (Machine Learning, Computer Vision, and Software Engineering roles).

---

## 🎯 30-Second Elevator Pitch

> *"I engineered **SignBridge**, a real-time assistive translation system that bridges communication between the hearing/speech-impaired community and non-signers. It tracks 3D hand skeletal keypoints using MediaPipe, applies spatial coordinate normalization to achieve scale and position invariance, classifies Indian Sign Language (ISL) gestures via a deep Keras MLP, and accumulates signs into punctuated sentences via temporal smoothing. The output is synthesized into natural spoken audio via an asynchronous multi-backend TTS engine running at 30+ FPS on a CPU without requiring a GPU."*

---

## 💡 Top Interview Questions & How to Answer Them

### 1. "Why did you use MediaPipe keypoints instead of feeding raw webcam frames into a 3D-CNN or ConvLSTM?"
**Your Answer:**
> *"Raw video frames contain massive spatial redundancy (background noise, varied lighting, skin tones, clothing). Passing 224x224 RGB frames into a 3D-CNN or Vision Transformer requires enormous compute (~tens of GFLOPs) and typically demands a high-end GPU to achieve real-time rates.*  
> *By decoupling feature representation (extracting 21 3D landmarks via MediaPipe) from classification, we compress each frame down to an ultra-compact vector (63 features). This reduced inference latency to **under 12 milliseconds**, made the model robust across lighting conditions and skin tones, and allowed real-time 30 FPS execution on budget consumer CPUs."*

### 2. "How did you handle the fact that people stand at different distances or positions from the camera?"
**Your Answer:**
> *"MediaPipe returns coordinates normalized to screen resolution $[0, 1]$, but they are still sensitive to where the hand is located in the frame and how close it is to the lens.*  
> *To solve this, I designed a two-step spatial normalization pipeline:*  
> 1. ***Translation Invariance:*** *Shifted all 21 landmarks relative to the wrist $(p_0)$, centering the wrist at origin $(0, 0, 0)$.*  
> 2. ***Scale Invariance:*** *Normalized all coordinate vectors by dividing by the Euclidean distance between the wrist and the middle finger MCP joint ($p_9$).*  
> *This guarantees that the input distribution fed to the neural network is completely invariant to hand placement and camera distance."*

### 3. "How did you eliminate frame flicker and false classifications during gesture transitions?"
**Your Answer:**
> *"Static frame-by-frame inference suffers from transition jitter when moving the hand between signs. I introduced two temporal layers in the `SentenceBuilder` module:*  
> 1. ***Sliding-Window Majority Voting:*** *A rolling FIFO buffer of recent frames $(N=7)$ filters out isolated outlier misclassifications.*  
> 2. ***Hold-Duration Commitment:*** *A gesture must remain stable for at least 5 consecutive frames before it is committed as a word token.*  
> *This eliminated single-frame flicker and enabled smooth word-by-word sentence construction."*

### 4. "How did you prevent the audio playback from stuttering or freezing the video stream?"
**Your Answer:**
> *"Neural vocoders and speech engines like Tacotron2/HiFi-GAN or pyttsx3 are computationally heavy or execute blocking I/O calls. If called directly within the OpenCV or Tkinter main loop, the video frame rate drops to zero during speech.*  
> *I decoupled speech synthesis by building a producer-consumer architecture using a thread-safe `queue.Queue` and a background daemon worker thread. The recognition loop simply enqueues text tokens, while the worker synthesizes and plays audio independently."*

---

## 💼 Resume Formats

### For Machine Learning / Computer Vision Engineer:
- **SignBridge — Real-Time Indian Sign Language Translator** `[Python, OpenCV, MediaPipe, TensorFlow/Keras, NumPy]`
  - Engineered an end-to-end multi-modal ISL recognition pipeline achieving **30+ FPS** with **<12ms latency** on CPU-only hardware.
  - Formulated origin-centered, palm-span scale normalization on 21 3D hand landmarks, achieving position and distance invariance.
  - Implemented temporal sliding-window majority voting and hold-duration thresholds to filter transition-state noise.
  - Trained and evaluated an optimized Keras MLP with Batch Normalization and Dropout, exporting models to TFLite for edge efficiency.

### For Full-Stack / Software Engineer:
- **SignBridge — Assistive Voice & Gesture Translation Platform** `[Python, Multithreading, Tkinter, OpenCV, pyttsx3]`
  - Architected a modular multi-threaded desktop application integrating computer vision tracking with asynchronous voice synthesis.
  - Designed a non-blocking producer-consumer queue to decouple video ingestion, neural inference, and audio playback.
  - Built an interactive dataset collection studio with on-screen HUD, speeding up new gesture onboarding to under 2 minutes.
  - Structured production codebase with automated test suites (`pytest`), strict PEP8 modularity, and cross-platform launcher scripts.

---

## 🎬 Live Demo Checklist (For Presentations / Viva / Interviews)

1. **Setup**: Run `python app.py` (or double-click `run.bat`).
2. **Step 1 - Show Live Skeletal Tracking**: Point hand at camera, toggle the "Hand Skeleton Overlay" checkmark to show the 21 3D landmarks moving in real-time.
3. **Step 2 - Demonstrate Confidence Gauge**: Form signs (`Hello`, `Good`, `Morning`, `Sorry`) and show how the confidence bar reacts instantly.
4. **Step 3 - Demonstrate Sentence Accumulator**:
   - Sign `Hello` (hold until the hold-bar reaches 100% and commits).
   - Sign `Good`.
   - Sign `Morning`.
   - Show how the text box automatically forms: *"Hello, good morning."*
5. **Step 4 - Audio Output**: Click **"🔊 Speak Sentence"** to hear the asynchronous TTS engine speak the sentence out loud.
6. **Step 5 - Show Undo/Clear**: Click `⌫ Undo` to delete the last sign and `🗑 Clear` to reset.
7. **Step 6 - Dataset Studio**: Mention how `python train.py` enables users to record and train custom gestures in 60 seconds.
