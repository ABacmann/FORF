from abc import ABC, abstractmethod
from PIL import Image


# Define the strategy interface
class InpaintingModel(ABC):
    @abstractmethod
    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        pass
