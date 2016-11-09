# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import argrelextrema
import itertools

class CharSegmentation(object):

    def __init__(self, num_char, debug_path = None):
        self.num_char = num_char
        self.debug_path = debug_path
        self.bin_img = None

    def find_lowest_nonzero_curve(self, bin_img):
        #cv2.imshow('bin_img', bin_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        height, width = bin_img.shape
        nonzero_curve = []
        for i in range(width):
            is_found = False
            for j in range(height):
                if bin_img[j, i] != 0:
                    nonzero_curve.append(j)
                    is_found = True
                    break
            if not is_found:
                nonzero_curve.append(height-1)
        return np.asarray(nonzero_curve)

    def merge_closest_points(self, min_x_coordinates, min_diff_x=10):
        ret = []
        n = len(min_x_coordinates)
        taken = [False] * n
        for i in range(n):
            if not taken[i]:
                count = 1
                point = min_x_coordinates[i]
                taken[i] = True
                for j in range(i+1, n):
                    if abs(min_x_coordinates[i] - min_x_coordinates[j]) < min_diff_x:
                        point += min_x_coordinates[j]
                        count+=1
                        taken[j] = True
                point /= count
                ret.append(point)
        return ret

    def remove_noise_by_contours(self, bin_img):
        c_bin_img = np.copy(bin_img)
        min_area = 100
        max_area = bin_img.shape[0] * bin_img.shape[1]
        min_w = 10
        min_h = 10
        if cv2.__version__[0] == "2":
            contours, hierarchy = cv2.findContours(
                c_bin_img,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, hierarchy = cv2.findContours(
                c_bin_img,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area and (h >= min_h \
                    or w >= min_w) and w * h <= max_area:
                filtered_contours.append(cnt)
            else:
                bin_img[y:y+h, x:x+w] = 0
        contours = filtered_contours
        return bin_img


    def do(self, cv2_color_img):
        # return all the possible segmentations
        cv_grey_img = cv2.cvtColor(cv2_color_img, cv2.COLOR_BGR2GRAY)
        height, width = cv_grey_img.shape
        adaptive_threshold = cv2.adaptiveThreshold(
            cv_grey_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV, 11, 2)

        adaptive_threshold = self.remove_noise_by_contours(adaptive_threshold)
        self.bin_img = adaptive_threshold

        nonzero_curve = self.find_lowest_nonzero_curve(adaptive_threshold)
        nonzero_curve = savgol_filter(nonzero_curve, 15, 2)
        min_points = argrelextrema(nonzero_curve, np.greater)
        min_points = min_points[0]
        min_points = [i for i in min_points]

        #min_points.append(width-1)
        #min_points = [0,] + min_points

        min_points = self.merge_closest_points(min_points, width * 0.02)

        print("min_points=", min_points)
        segmentations = []
        for selected_min_points in itertools.combinations(min_points,
                                                          self.num_char+1):
            print("selected_min_points=", selected_min_points)
            one_segmentation = []
            prev_min_point = selected_min_points[0]
            for i, selected_min_point in enumerate(selected_min_points):
                if i != 0:
                    one_segmentation.append(
                        (prev_min_point, 0,
                         selected_min_point - prev_min_point, height))
                    prev_min_point = selected_min_point
            segmentations.append(one_segmentation)

        if self.debug_path is not None:
            import matplotlib.pyplot as plt
            path_cv2_color_img = os.path.join(self.debug_path,
                                              "cv2_color_img.jpg")
            path_cv_grey_img = os.path.join(self.debug_path,
                                              "cv2_grey_img.jpg")
            path_adaptive_threshold = os.path.join(self.debug_path,
                                              "adaptive_threshold.jpg")
            ## draw possible segmentation on image
            for min_point in min_points:
                cv2.line(cv2_color_img, (min_point, 0),
                         (min_point, height), (255, 0, 0))

            cv2.imwrite(path_cv2_color_img, cv2_color_img)
            cv2.imwrite(path_cv_grey_img, cv_grey_img)
            cv2.imwrite(path_adaptive_threshold, adaptive_threshold)

            min_point_vals = [nonzero_curve[i] for i in min_points]
            #plt.plot(range(nonzero_curve.shape[0]), nonzero_curve)
            #plt.plot(min_points, min_point_vals, 'ro')
            #plt.gca().invert_yaxis()
            #plt.show()
        return segmentations
