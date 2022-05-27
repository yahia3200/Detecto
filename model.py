from PIL import Image, ImageEnhance
import cv2
import numpy as np
from tqdm import tqdm


class DataLoader():

    def __init__(self) -> None:
        pass

    def transform(self, imgs_path, **transform_params):
        return [Image.open(img_path) for img_path in imgs_path]

    def fit(self, X, y=None, **fit_params):
        return self


class PreProcessor():

    def __init__(self) -> None:
        pass

    def preprocess_image(self, im, sharpness_factor=10, bordersize=3):

        bright = ImageEnhance.Brightness(im)
        if(np.average(np.array(im)) < 128):
            im = bright.enhance(2.5)

        enhancer = ImageEnhance.Sharpness(im)
        im_s_1 = enhancer.enhance(sharpness_factor)
        image = np.array(im_s_1)
        image = cv2.copyMakeBorder(
            image,
            top=bordersize,
            bottom=bordersize,
            left=bordersize,
            right=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255]
        )
        orig_image = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)

        (thresh, bw_image) = cv2.threshold(
            image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return bw_image, orig_image

    def get_contour_pixels(self, bw_image):
        contours, _ = cv2.findContours(
            bw_image, cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
        return contours

    def transform(self, imgs, **transform_params):
        preprocessed_imgs = []
        for i in range(len(imgs)):
            preprocessed_imgs.append(
                self.get_contour_pixels(self.preprocess_image(imgs[i])[0]))
        return preprocessed_imgs

    def fit(self, X, y=None, **fit_params):
        return self


class FeatureExtractor():
    def __init__(self) -> None:
        pass

    def get_hinge_features(self, contours):

        N_ANGLE_BINS = 40
        BIN_SIZE = 360 // N_ANGLE_BINS
        LEG_LENGTH = 25

        hist = np.zeros((N_ANGLE_BINS, N_ANGLE_BINS))

        # print([len(cnt) for cnt in contours])
        for cnt in contours:
            n_pixels = len(cnt)
            if n_pixels <= LEG_LENGTH:
                continue

            points = np.array([point[0] for point in cnt])
            xs, ys = points[:, 0], points[:, 1]
            point_1s = np.array([cnt[(i + LEG_LENGTH) % n_pixels][0]
                                for i in range(n_pixels)])
            point_2s = np.array([cnt[(i - LEG_LENGTH) % n_pixels][0]
                                for i in range(n_pixels)])
            x1s, y1s = point_1s[:, 0], point_1s[:, 1]
            x2s, y2s = point_2s[:, 0], point_2s[:, 1]

            phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
            phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)

            indices = np.where(phi_2s > phi_1s)[0]

            for i in indices:
                phi1 = int(phi_1s[i] // BIN_SIZE) % N_ANGLE_BINS
                phi2 = int(phi_2s[i] // BIN_SIZE) % N_ANGLE_BINS
                hist[phi1, phi2] += 1

        normalised_hist = hist / np.sum(hist)
        feature_vector = normalised_hist[np.triu_indices_from(
            normalised_hist, k=1)]

        return feature_vector

    def transform(self, contours, **transform_params):
        X = [self.get_hinge_features(contour) for contour in contours]
        return X

    def fit(self, X, y=None, **fit_params):
        return self
