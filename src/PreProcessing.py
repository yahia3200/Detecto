import cv2

class PreProcessor():

    def __init__(self) -> None:
        pass

    def preprocess_image(self, im, sharpness_factor=10, bordersize=3):

        image = im
        image[-500:, :] = 130

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (3, 3), 0)

        (thresh, bw_image) = cv2.threshold(
            image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return bw_image, im

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
