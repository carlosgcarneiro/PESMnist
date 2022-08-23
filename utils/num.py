import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


class Num:
    
    def __init__(self, path: str, invert: bool = False) -> None:
        self.img_path = path
        self.img = np.asarray(Image.open(path))

        if invert:
            self.img = np.invert(self.img)
        
    def plot(self) -> None:
        plt.imshow(self.img, cmap='gray', vmin=0, vmax=255)
        plt.show()
    
    def to_array(self) -> np.array:
        return np.reshape(self.img, (1,-1))    