"""
Unit & Integration Tests for ISL Recognition Pipeline.
Run with: pytest tests/
"""
import unittest
import numpy as np
from src.core.sentence_builder import SentenceBuilder


class TestSentenceBuilder(unittest.TestCase):
    """Test temporal smoothing, hold duration, and sentence accumulation."""

    def setUp(self):
        self.builder = SentenceBuilder(
            confidence_threshold=0.70,
            window_size=5,
            hold_frames_required=3,
            cooldown_seconds=0.1
        )

    def test_empty_initial_state(self):
        self.assertEqual(self.builder.get_sentence(), "")
        self.assertEqual(len(self.builder.get_words()), 0)

    def test_low_confidence_rejection(self):
        # Predictions below threshold should not commit
        for _ in range(5):
            smoothed, hold_progress, committed = self.builder.update("Hello", 0.50)
            self.assertIsNone(committed)
        self.assertEqual(len(self.builder.get_words()), 0)

    def test_hold_and_commit(self):
        # Hold "Hello" for enough frames to establish window majority and hold duration
        committed_word = None
        for _ in range(6):
            smoothed, hold_progress, newly_committed = self.builder.update("Hello", 0.95)
            if newly_committed:
                committed_word = newly_committed

        self.assertEqual(committed_word, "Hello")
        self.assertEqual(self.builder.get_sentence(), "Hello.")

    def test_sentence_accumulation(self):
        # Commit "Hello"
        for _ in range(5):
            self.builder.update("Hello", 0.95)

        # Hand drops between gestures
        self.builder.update(None, 0.0)

        # Wait out cooldown
        import time
        time.sleep(0.15)

        # Commit "Morning"
        for _ in range(5):
            self.builder.update("Morning", 0.95)

        self.assertEqual(self.builder.get_words(), ["Hello", "Morning"])
        self.assertEqual(self.builder.get_sentence(), "Hello morning.")

    def test_backspace_and_clear(self):
        self.builder.words = ["Good", "Morning"]
        removed = self.builder.backspace()
        self.assertEqual(removed, "Morning")
        self.assertEqual(self.builder.get_words(), ["Good"])

        self.builder.clear()
        self.assertEqual(self.builder.get_words(), [])
        self.assertEqual(self.builder.get_sentence(), "")


if __name__ == "__main__":
    unittest.main()
