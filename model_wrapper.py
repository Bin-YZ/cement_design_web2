"""
Model wrapper for loading and running trained Keras models.
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import warnings


class ModelWrapper:
    """Wrapper for loading and running the trained Keras model."""
    
    FEATURES = [
        'C3S', 'C2S', 'C3A', 'time', 'C4AF',
        'silica_fume', 'GGBFS', 'fly_ash', 'calcined_clay', 'limestone'
    ]
    
    @staticmethod
    def _weighted_mse(y_true, y_pred):
        """Custom loss function used when loading the model."""
        e_err = tf.square(y_true[:, 0] - y_pred[:, 0])
        co2_err = tf.square(y_true[:, 1] - y_pred[:, 1])
        return tf.reduce_mean(e_err + co2_err)
    
    def __init__(self, model_path: str, suppress_warnings: bool = True):
        """Load model with custom loss.
        
        Args:
            model_path: Path to the model file
            suppress_warnings: If True, suppress TensorFlow warnings
        """
        self.model_path = model_path
        
        if suppress_warnings:
            # Temporarily suppress warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.model = load_model(
                    model_path,
                    custom_objects={"weighted_mse": ModelWrapper._weighted_mse}
                )
                
                # Trigger metrics initialization with a dummy prediction
                # This eliminates the "compiled metrics have yet to be built" warning
                dummy_input = np.zeros((1, len(self.FEATURES)), dtype='float32')
                _ = self.model.predict(dummy_input, verbose=0)
        else:
            self.model = load_model(
                model_path,
                custom_objects={"weighted_mse": ModelWrapper._weighted_mse}
            )
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run predictions on a DataFrame of input features.
        
        Returns:
            Array with columns: [E, CO2_abs]
        """
        X = df[ModelWrapper.FEATURES].values.astype("float32")
        y = self.model.predict(X, verbose=0)
        return y
    
    def __getstate__(self):
        """Custom pickle support - don't pickle the model itself."""
        state = self.__dict__.copy()
        # Remove the unpicklable model
        state['model'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickle support - reload the model."""
        self.__dict__.update(state)
        # Reload the model (with warnings suppressed)
        if self.model_path:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                self.model = load_model(
                    self.model_path,
                    custom_objects={"weighted_mse": ModelWrapper._weighted_mse}
                )
                # Trigger metrics initialization
                dummy_input = np.zeros((1, len(self.FEATURES)), dtype='float32')
                _ = self.model.predict(dummy_input, verbose=0)
