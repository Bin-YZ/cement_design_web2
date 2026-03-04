
from optimizer_gui import OptimizerGUI


def create_gui(model_path: str) -> OptimizerGUI:
    """Create and display the optimization GUI.
    
    Args:
        model_path: Path to trained Keras model file
        
    Returns:
        OptimizerGUI instance
    """
    return OptimizerGUI(model_path)


if __name__ == "__main__":
    # Example usage
    MODEL_PATH = "my_model16082025.h5"  # Update with your model path
    gui = create_gui(MODEL_PATH)
