"""
Temporal smoothing and sentence construction engine.
Eliminates single-frame classification jitter using majority voting,
enforces hold durations for deliberate signs, and assembles words into punctuated sentences.
"""
from collections import deque, Counter
import time


class SentenceBuilder:
    """Manages gesture smoothing, hold-confirmation, and sentence accumulation."""

    def __init__(self,
                 confidence_threshold=0.70,
                 window_size=7,
                 hold_frames_required=5,
                 cooldown_seconds=1.2):
        """
        Initialize the sentence builder.
        
        Args:
            confidence_threshold: Minimum probability to consider a prediction.
            window_size: Number of recent frames to use for majority voting.
            hold_frames_required: Number of consecutive frames the same smoothed gesture
                                  must be held to trigger word commitment.
            cooldown_seconds: Cooldown to prevent repeating the exact same word unintentionally.
        """
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.hold_frames_required = hold_frames_required
        self.cooldown_seconds = cooldown_seconds

        # Buffers
        self.frame_buffer = deque(maxlen=window_size)
        self.words = []

        # State tracking
        self.current_candidate = None
        self.candidate_hold_count = 0
        self.last_committed_word = None
        self.last_committed_time = 0

    def update(self, raw_label, confidence):
        """
        Feed a raw frame prediction and update the smoothing/sentence state.
        
        Args:
            raw_label: Predicted label string or None if no hand detected.
            confidence: Probability score in [0.0, 1.0].
            
        Returns:
            Tuple of:
                smoothed_label (str or None): Currently dominating gesture.
                hold_progress (float): Fraction [0.0, 1.0] towards committing the gesture.
                newly_committed_word (str or None): Word emitted in this frame, if any.
        """
        newly_committed = None

        # Filter low confidence or missing detections
        if raw_label is None or confidence < self.confidence_threshold:
            self.frame_buffer.append(None)
        else:
            self.frame_buffer.append(raw_label)

        # Majority vote across window
        valid_predictions = [lbl for lbl in self.frame_buffer if lbl is not None]
        if not valid_predictions:
            self.current_candidate = None
            self.candidate_hold_count = 0
            return None, 0.0, None

        counter = Counter(valid_predictions)
        smoothed_label, count = counter.most_common(1)[0]

        # Require majority in window
        if count < (self.window_size // 2 + 1):
            return None, 0.0, None

        now = time.time()

        # Check if candidate matches current held gesture
        if smoothed_label == self.current_candidate:
            self.candidate_hold_count += 1
        else:
            self.current_candidate = smoothed_label
            self.candidate_hold_count = 1

        hold_progress = min(1.0, self.candidate_hold_count / float(self.hold_frames_required))

        # Check if held long enough to commit
        if self.candidate_hold_count >= self.hold_frames_required:
            is_same_as_last = (smoothed_label == self.last_committed_word)
            time_since_last = now - self.last_committed_time

            # Only commit if it's a new word or cooldown expired
            if not is_same_as_last or time_since_last > self.cooldown_seconds:
                self.words.append(smoothed_label)
                newly_committed = smoothed_label
                self.last_committed_word = smoothed_label
                self.last_committed_time = now
                self.candidate_hold_count = 0  # Reset after commit

        return smoothed_label, hold_progress, newly_committed

    def backspace(self):
        """Remove the most recently added word."""
        if self.words:
            removed = self.words.pop()
            return removed
        return None

    def clear(self):
        """Clear the entire sentence buffer."""
        self.words.clear()
        self.current_candidate = None
        self.candidate_hold_count = 0
        self.last_committed_word = None

    def get_sentence(self):
        """Return the formatted sentence string with proper grammar capitalization."""
        if not self.words:
            return ""

        raw_text = " ".join(self.words).strip()
        if not raw_text:
            return ""

        # Format first letter uppercase, remaining lowercase, and end with period
        formatted = raw_text[0].upper() + raw_text[1:].lower()
        if not formatted.endswith((".", "!", "?")):
            formatted += "."
        return formatted

    def get_words(self):
        """Return raw list of word tokens in the sentence."""
        return list(self.words)
