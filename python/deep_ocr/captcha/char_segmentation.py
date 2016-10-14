# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

class CharSegmentation(object):
    
    def __init__(self, num_char, debug_path = None):
        self.num_char = num_char
        self.debug_path = debug_path
    
    def do(self, cv2_color_img):
        cv_grey_img = cv2.cvtColor(cv2_color_img, cv2.COLOR_BGR2GRAY)
        adaptive_threshold = cv2.adaptiveThreshold(
            cv_grey_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 11, 2)

        vertical_sum = np.sum(adaptive_threshold, axis=0)

        if self.debug_path is not None:
            import matplotlib.pyplot as plt
            path_cv2_color_img = os.path.join(self.debug_path,
                                              "cv2_color_img.jpg")
            path_cv_grey_img = os.path.join(self.debug_path,
                                              "cv2_grey_img.jpg")
            path_adaptive_threshold = os.path.join(self.debug_path,
                                              "adaptive_threshold.jpg")
            cv2.imwrite(path_cv2_color_img, cv2_color_img)
            cv2.imwrite(path_cv_grey_img, cv_grey_img)
            cv2.imwrite(path_adaptive_threshold, adaptive_threshold)

            plt.plot(range(vertical_sum.shape[0]), vertical_sum)
            plt.show()
