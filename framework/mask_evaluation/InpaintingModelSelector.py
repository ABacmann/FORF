import cv2
import numpy as np

from framework.mask_evaluation.Mask_Analyzer import MaskAnalyzer
from framework.mask_evaluation.ModelPerformaceDatabase import ModelPerformanceDatabase


class InpaintingModelSelector:
    def __init__(self):
        self.model_db = ModelPerformanceDatabase()

    def select_model(self, mask: np.ndarray):
        analyzer = MaskAnalyzer(mask)
        features = analyzer.get_features()

        # Normalize or weight features as needed
        weighted_features = {
            'area': features['area'] / 10000,  # Example normalization
            'aspect_ratio': features['aspect_ratio'],
            'complexity': features['complexity']
        }

        best_model = self.model_db.get_best_model(weighted_features)
        return best_model

# Usage example
if __name__ == "__main__":
    # Example usage with a sample mask image
    mask = cv2.imread('path_to_mask.png', cv2.IMREAD_GRAYSCALE)
    selector = InpaintingModelSelector()
    recommended_model = selector.select_model(mask)
    print(f"Recommended model for inpainting: {recommended_model}")
