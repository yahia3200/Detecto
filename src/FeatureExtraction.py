import cv2
import numpy as np


class FeatureExtractor():
    def __init__(self) -> None:
        pass

    def get_hinge_features(self, contours):

        N_ANGLE_BINS = 40
        BIN_SIZE = 360 // N_ANGLE_BINS
        LEG_LENGTH = 25

        hist = np.zeros((N_ANGLE_BINS, N_ANGLE_BINS))

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

    def get_cold_features(self, contours, approx_poly_factor=0.01):

        N_RHO_BINS = 7
        N_ANGLE_BINS = 12
        N_BINS = N_RHO_BINS * N_ANGLE_BINS
        BIN_SIZE = 360 // N_ANGLE_BINS
        R_INNER = 5.0
        R_OUTER = 35.0
        K_S = np.arange(3, 8)

        rho_bins_edges = np.log10(np.linspace(R_INNER, R_OUTER, N_RHO_BINS))
        feature_vectors = np.zeros((len(K_S), N_BINS))

        # print([len(cnt) for cnt in contours])
        for j, k in enumerate(K_S):
            hist = np.zeros((N_RHO_BINS, N_ANGLE_BINS))
            for cnt in contours:
                epsilon = approx_poly_factor * cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, epsilon, True)
                n_pixels = len(cnt)

                point_1s = np.array([point[0] for point in cnt])
                x1s, y1s = point_1s[:, 0], point_1s[:, 1]
                point_2s = np.array([cnt[(i + k) % n_pixels][0]
                                    for i in range(n_pixels)])
                x2s, y2s = point_2s[:, 0], point_2s[:, 1]

                thetas = np.degrees(np.arctan2(y2s - y1s, x2s - x1s) + np.pi)
                rhos = np.sqrt((y2s - y1s) ** 2 + (x2s - x1s) ** 2)
                rhos_log_space = np.log10(rhos)

                quantized_rhos = np.zeros(rhos.shape, dtype=int)
                for i in range(N_RHO_BINS):
                    quantized_rhos += (rhos_log_space < rho_bins_edges[i])

                for i, r_bin in enumerate(quantized_rhos):
                    theta_bin = int(thetas[i] // BIN_SIZE) % N_ANGLE_BINS
                    hist[r_bin - 1, theta_bin] += 1

            normalised_hist = hist / hist.sum()
            feature_vectors[j] = normalised_hist.flatten()

        return feature_vectors.flatten()

    #np.concatenate([self.get_hinge_features(contour), self.get_cold_features(contour)])
    def transform(self, contours, **transform_params):
        X = [self.get_hinge_features(contour) for contour in contours]
        return X

    def fit(self, X, y=None, **fit_params):
        return self
