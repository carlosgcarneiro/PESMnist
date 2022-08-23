import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class TestNum:
    
    def __init__(self, path: str) -> None:
        self.img_path = path
        self.img = np.asarray(Image.open(path))
        
    def print_number(self, invert: bool = False) -> None:
        if invert:
            img = np.invert(img)
        
        plt.imshow(self.img, cmap='gray', vmin=0, vmax=255)
        plt.show()
        