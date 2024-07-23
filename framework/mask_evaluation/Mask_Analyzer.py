import cv2
import numpy as np


class MaskAnalyzer:
    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.features = self.analyze_mask()

    def analyze_mask(self):
        # Ensure the mask is binary
        _, binary_mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)

        # Calculate mask area
        area = cv2.countNonZero(binary_mask)

        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(binary_mask)
        aspect_ratio = w / float(h) if h != 0 else 0

        # Calculate shape complexity (contour length / area)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            complexity = perimeter / float(area) if area != 0 else 0
        else:
            perimeter = 0
            complexity = 0

        # Extract additional features as needed
        features = {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'perimeter': perimeter,
            'complexity': complexity,
            'bounding_box': (x, y, w, h)
        }

        return features

    def get_features(self):
        return self.features


# Usage example
if __name__ == "__main__":
    # Example usage with a sample mask image
    mask = cv2.imread('path_to_mask.png', cv2.IMREAD_GRAYSCALE)
    analyzer = MaskAnalyzer(mask)
    features = analyzer.get_features()
    print(features)
