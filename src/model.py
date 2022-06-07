from PIL import Image, ImageEnhance
import cv2
import numpy as np
from tqdm import tqdm


class DataLoader():

    def __init__(self) -> None:
        pass

    def transform(self, imgs_path, **transform_params):
        return [cv2.imread(img_path) for img_path in imgs_path]

    def fit(self, X, y=None, **fit_params):
        return self
