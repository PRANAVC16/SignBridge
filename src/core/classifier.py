"""
Neural Network classifier for Sign Language gesture recognition.
Wraps TensorFlow / Keras model definition, training, inference, and serialization.
"""
import os
import json
from pathlib import Path
import numpy as np

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class SignLanguageClassifier:
    """Manages training, inference, and serialization of sign language classification models."""

    def __init__(self, model_path=None, labels_path=None):
        """
        Initialize classifier.
        
        Args:
            model_path: Optional path to pre-trained .h5 or .keras model file.
            labels_path: Optional path to gesture_labels.json.
        """
        self.model = None
        self.labels = {}  # {str(idx): label_name}
        self.labels_path = labels_path
        self.model_path = model_path

        if labels_path and os.path.exists(labels_path):
            self.load_labels(labels_path)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_labels(self, path):
        """Load class index-to-label mapping from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            raw_labels = json.load(f)
            # Ensure keys are strings for JSON consistency
            self.labels = {str(k): str(v) for k, v in raw_labels.items()}
        self.labels_path = path
        return self.labels

    def save_labels(self, path=None):
        """Save label mapping to JSON file."""
        target_path = path or self.labels_path
        if not target_path:
            raise ValueError("No destination path specified for saving labels.")

        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)

    def build_model(self, input_dim=63, num_classes=4, learning_rate=1e-3):
        """
        Build an optimized Deep Multi-Layer Perceptron (MLP) architecture.
        
        Args:
            input_dim: Number of input features (63 for 1-hand, 126 for 2-hands).
            num_classes: Number of distinct gesture categories.
            learning_rate: Adam optimizer learning rate.
            
        Returns:
            Compiled tf.keras.Model.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to build models.")

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
        return model

    def load_model(self, model_path):
        """Load trained Keras model from disk."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to load models.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = tf.keras.models.load_model(model_path)
        self.model_path = model_path
        return self.model

    def save_model(self, model_path=None):
        """Save Keras model to disk."""
        target_path = model_path or self.model_path
        if not target_path:
            raise ValueError("No destination path specified for saving model.")

        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(target_path)
        self.model_path = target_path

    def predict(self, features):
        """
        Predict gesture class from input features.
        
        Args:
            features: 1D np.ndarray of shape (input_dim,).
            
        Returns:
            Tuple of:
                predicted_idx (int): Highest probability class index.
                label (str): Human-readable gesture label.
                confidence (float): Probability score in [0.0, 1.0].
                probabilities (np.ndarray): Probability distribution over all classes.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded or initialized.")

        features_input = np.array(features, dtype=np.float32).reshape(1, -1)
        probabilities = self.model.predict(features_input, verbose=0)[0]
        predicted_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_idx])
        label = self.labels.get(str(predicted_idx), f"Class {predicted_idx}")

        return predicted_idx, label, confidence, probabilities

    def train(self, X, y, epochs=35, batch_size=16, val_split=0.15, callbacks=None):
        """
        Train or fine-tune model on captured gesture data.
        
        Args:
            X: np.ndarray of shape (N, input_dim).
            y: np.ndarray of shape (N,).
            epochs: Max training epochs.
            batch_size: Batch size.
            val_split: Fraction of samples for validation.
            callbacks: Optional list of Keras callbacks.
            
        Returns:
            history: Keras Training history dict.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to train models.")

        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        input_dim = X.shape[1]

        if self.model is None or self.model.output_shape[-1] != num_classes:
            self.build_model(input_dim=input_dim, num_classes=num_classes)

        default_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_split > 0 else 'loss',
                patience=8,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_split > 0 else 'loss',
                factor=0.5,
                patience=4
            )
        ]
        active_callbacks = callbacks if callbacks is not None else default_callbacks

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=active_callbacks,
            verbose=1
        )
        return history

    def export_tflite(self, output_path):
        """
        Export trained model to TensorFlow Lite (.tflite) for edge and embedded deployment.
        
        Args:
            output_path: Target path for the .tflite file.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required to export TFLite.")
        if self.model is None:
            raise RuntimeError("No model loaded to export.")

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        return output_path
