"""
Hand landmark extraction and spatial normalization using MediaPipe Hands.
Supports both single-hand and dual-hand tracking, with raw and invariant normalized coordinate modes.
"""
import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


class LandmarkExtractor:
    """Extracts, normalizes, and visualizes hand landmarks from video frames."""

    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.7):
        """Initialize MediaPipe Hands solution."""
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe is not installed. Run 'pip install mediapipe'.")

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame_bgr):
        """
        Process a BGR frame with MediaPipe Hands.
        
        Args:
            frame_bgr: OpenCV BGR image numpy array.
            
        Returns:
            results: MediaPipe Hands detection results object.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        return results

    def extract_features(self, results, mode="single", normalize=False):
        """
        Extract feature vector from detection results.
        
        Args:
            results: MediaPipe Hands results object.
            mode: "single" (63 features) or "dual" (126 features).
            normalize: If True, centers landmarks on wrist and scales by hand size.
                       If False, preserves raw [x, y, z] for legacy model compatibility.
                       
        Returns:
            features: np.ndarray of shape (63,) or (126,), or None if no hand detected.
        """
        if not results.multi_hand_landmarks:
            return None

        if mode == "single":
            # Extract primary hand (first detected hand)
            hand_landmarks = results.multi_hand_landmarks[0]
            raw_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

            if normalize:
                # 1. Translation Invariance: Shift origin to wrist (landmark 0)
                wrist = raw_coords[0, :]
                centered = raw_coords - wrist

                # 2. Scale Invariance: Divide by distance between wrist (0) and middle MCP (9)
                scale = np.linalg.norm(centered[9, :])
                if scale < 1e-5:
                    scale = 1.0
                normalized = centered / scale
                return normalized.flatten()
            else:
                # Raw flattened coordinates (21 * 3 = 63)
                return raw_coords.flatten()

        elif mode == "dual":
            # Dual hand extraction: up to 2 hands (21 * 3 * 2 = 126 features)
            dual_features = np.zeros((42, 3), dtype=np.float32)

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
                if normalize:
                    wrist = coords[0, :]
                    centered = coords - wrist
                    scale = np.linalg.norm(centered[9, :])
                    if scale < 1e-5:
                        scale = 1.0
                    coords = centered / scale

                start_idx = idx * 21
                end_idx = start_idx + 21
                dual_features[start_idx:end_idx] = coords

            return dual_features.flatten()

        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'single' or 'dual'.")

    def draw_landmarks(self, frame_bgr, results, draw_connections=True):
        """
        Draw hand skeleton and keypoints onto the frame in-place.
        
        Args:
            frame_bgr: OpenCV image to draw on.
            results: MediaPipe results object.
            draw_connections: Whether to draw connecting lines between joints.
        """
        if not results.multi_hand_landmarks:
            return

        for hand_landmarks in results.multi_hand_landmarks:
            if draw_connections:
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            else:
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    None
                )

    def get_bounding_boxes(self, results, frame_shape):
        """
        Compute bounding boxes around detected hands in pixel coordinates.
        
        Args:
            results: MediaPipe results.
            frame_shape: (height, width, channels)
            
        Returns:
            boxes: List of (x_min, y_min, x_max, y_max) tuples.
        """
        boxes = []
        if not results.multi_hand_landmarks:
            return boxes

        h, w = frame_shape[:2]
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]

            x_min = max(0, int(min(xs)) - 20)
            y_min = max(0, int(min(ys)) - 20)
            x_max = min(w, int(max(xs)) + 20)
            y_max = min(h, int(max(ys)) + 20)
            boxes.append((x_min, y_min, x_max, y_max))

        return boxes

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
