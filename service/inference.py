import numpy as np
from typing import List

class ModelInference:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Mock inference endpoint returning predictions."""
        batch_size = input_data.shape[0]
        return np.random.rand(batch_size, 10)  # 10 classes

inference_model = ModelInference()

def get_predictions(features: List[float]) -> List[float]:
    data = np.array(features).reshape(1, -1)
    preds = inference_model.predict(data)
    return preds[0].tolist()
