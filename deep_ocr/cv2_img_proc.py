# -*- coding: utf-8 -*-

import cv2
import numpy as np

class PreprocessRemoveNonCharNoise(object):

    def __init__(self, char_width):
        self.min_w = char_width * 0.1
        self.min_h = char_width * 0.1

        self.min_area = char_width * char_width * 0.05

        self.max_area = char_width * char_width * 2.0

    def do(self, bin_img):
        
        tmp_bin_img = np.copy(bin_img)

        if cv2.__version__[0] == "2":
            contours, hierarchy = cv2.findContours(
                tmp_bin_img,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, hierarchy = cv2.findContours(
                tmp_bin_img,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > self.max_area or w * h < self.min_area:
                bin_img[y:y+h, x:x+w] = 0
        contours = filtered_contours

class PreprocessBackgroundMask():
    
    def __init__(self, boundary):
        self.boundary = boundary

    def do(self, image):
        (lower, upper) = self.boundary
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        mask = cv2.inRange(image, lower, upper)
        return mask


class PreprocessCropZeros(object):

    def __init__(self):
        pass

    def do(self, cv2_gray_img):
        height = cv2_gray_img.shape[0]
        width = cv2_gray_img.shape[1]

        v_sum = np.sum(cv2_gray_img, axis=0)
        h_sum = np.sum(cv2_gray_img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1

        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break

        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break

        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break

        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break
        if not (top < low and right > left):
            return cv2_gray_img

        return cv2_gray_img[top: low+1, left: right+1]



class FindImageBBox(object):
    def __init__(self, ):
        pass

    def do(self, img):
        height = img.shape[0]
        width = img.shape[1]
        v_sum = np.sum(img, axis=0)
        h_sum = np.sum(img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1
        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break
        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break
        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break
        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break
        return (left, top, right, low)



class PreprocessResizeKeepRatio(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def do(self, cv2_img):
        max_width = self.width
        max_height = self.height

        cur_height, cur_width = cv2_img.shape[:2]

        ratio_w = float(max_width)/float(cur_width)
        ratio_h = float(max_height)/float(cur_height)
        ratio = min(ratio_w, ratio_h)

        new_size = (min(int(cur_width*ratio), max_width),
                    min(int(cur_height*ratio), max_height))

        new_size = (max(new_size[0], 1),
                    max(new_size[1], 1),)

        resized_img = cv2.resize(cv2_img, new_size)
        return resized_img


class PreprocessResizeKeepRatioFillBG(object):

    def __init__(self, width, height,
                 fill_bg=False,
                 auto_avoid_fill_bg=True,
                 margin=None):
        self.width = width
        self.height = height
        self.fill_bg = fill_bg
        self.auto_avoid_fill_bg = auto_avoid_fill_bg
        self.margin = margin

    @classmethod
    def is_need_fill_bg(cls, cv2_img, th=0.5, max_val=255):
        image_shape = cv2_img.shape
        height, width = image_shape
        if height * 3 < width:
            return True
        if width * 3 < height:
            return True
        return False

    @classmethod
    def put_img_into_center(cls, img_large, img_small, ):
        width_large = img_large.shape[1]
        height_large = img_large.shape[0]

        width_small = img_small.shape[1]
        height_small = img_small.shape[0]

        if width_large < width_small:
            raise ValueError("width_large <= width_small")
        if height_large < height_small:
            raise ValueError("height_large <= height_small")

        start_width = (width_large - width_small) / 2
        start_height = (height_large - height_small) / 2

        img_large[start_height:start_height + height_small,
                  start_width:start_width + width_small] = img_small
        return img_large

    def do(self, cv2_img):

        if self.margin is not None:
            width_minus_margin = max(2, self.width - self.margin)
            height_minus_margin = max(2, self.height - self.margin)
        else:
            width_minus_margin = self.width
            height_minus_margin = self.height

        cur_height, cur_width = cv2_img.shape[:2]
        if len(cv2_img.shape) > 2:
            pix_dim = cv2_img.shape[2]
        else:
            pix_dim = None

        preprocess_resize_keep_ratio = PreprocessResizeKeepRatio(
            width_minus_margin,
            height_minus_margin)
        resized_cv2_img = preprocess_resize_keep_ratio.do(cv2_img)

        if self.auto_avoid_fill_bg:
            need_fill_bg = self.is_need_fill_bg(cv2_img)
            if not need_fill_bg:
                self.fill_bg = False
            else:
                self.fill_bg = True

        ## should skip horizontal stroke
        if not self.fill_bg:
            ret_img = cv2.resize(resized_cv2_img, (width_minus_margin,
                                                   height_minus_margin))
        else:
            if pix_dim is not None:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin),
                                    np.uint8)
            ret_img = self.put_img_into_center(norm_img, resized_cv2_img)

        if self.margin is not None:
            if pix_dim is not None:
                norm_img = np.zeros((self.height,
                                     self.width,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((self.height,
                                     self.width),
                                    np.uint8)
            ret_img = self.put_img_into_center(norm_img, ret_img)
        return ret_img