# -*- coding: utf-8 -*-

import numpy as np
import cv2
from deep_ocr.cv2_img_proc import PreprocessCropZeros
from deep_ocr.cv2_img_proc import PreprocessResizeKeepRatioFillBG
from deep_ocr.utils import extract_peek_ranges_from_array
from deep_ocr.utils import merge_peek_ranges_mini_non_digits

class RectImageClassifier(object):
    def __init__(self, caffe_cls, bin_image, char_set,
                 caffe_cls_width=64, caffe_cls_height=64):
        self.caffe_cls = caffe_cls
        self.bin_image = bin_image
        self.cache_res = {}
        self.char_set = char_set
        self.caffe_cls_width = caffe_cls_width
        self.caffe_cls_height = caffe_cls_height

    def _do(self, rects, boundary):
        rects_to_reco = []
        for rect in rects:
            key = (rect, boundary)
            if key not in self.cache_res:
                rects_to_reco.append(rect)
        image = self.bin_image
        char_imgs = []
        crop_zeros = PreprocessCropZeros()
        resize_keep_ratio = PreprocessResizeKeepRatioFillBG(
            self.caffe_cls_width, self.caffe_cls_height,
            fill_bg=False, margin=4)
        for i, rect in enumerate(rects_to_reco):
            x, y, w, h = rect
            char_img = image[y:y+h+1, x:x+w+1]
            char_img = crop_zeros.do(char_img)
            char_img = resize_keep_ratio.do(char_img)
            char_imgs.append(char_img)
        np_char_imgs = np.asarray(char_imgs)
        output_tag_to_max_proba = self.caffe_cls.predict_cv2_imgs(np_char_imgs)
        for i, item in enumerate(output_tag_to_max_proba):
            cur_rect = rects_to_reco[i]
            key = (cur_rect, boundary)
            if len(self.char_set["set"]) > 0:
                for char_p in item:
                    if char_p[0] in self.char_set["set"]:
                        self.cache_res[key] = char_p
                        break
            else:
                self.cache_res[key] = item[0]

    def do(self, rects, boundary):
        self._do(rects, boundary)
        ocr_res = ""
        for rect in rects:
            key = (rect, boundary)
            ocr_res += self.cache_res[key][0]
        return ocr_res

    def do_images_maxproba(self, rects, boundaries, bin_images):
        size = len(boundaries)
        ## generate cache
        for i in range(size):
            boundary = boundaries[i]
            bin_image = bin_images[i]
            self.bin_image = bin_image
            self._do(rects, boundary)

        mat_proba = []
        for rect in rects:
            row_probabilities = []
            for i in range(size):
                boundary = boundaries[i]
                key = (rect, boundary)
                row_probabilities.append(self.cache_res[key])
            mat_proba.append(row_probabilities)

        ocr_res = ""
        n = len(mat_proba)
        for i in range(n):
            mat_proba[i] = sorted(mat_proba[i], key=lambda x: -x[1])
            ocr_res += mat_proba[i][0][0]
        return ocr_res
    
class RecoTextLine(object):
    def __init__(self, rect_img_clf,
                 char_set=None,
                 debug_path=None):
        self.char_set = char_set
        self.debug_path = debug_path
        self.rect_img_clf = rect_img_clf


    def convert_peek_ranges_into_rects(self,
                                       peek_ranges,
                                       line_rect):
        base_x, base_y, base_w, base_h = line_rect
        rects = []
        for peek_range in peek_ranges:
            x = base_x + peek_range[0]
            y = base_y
            w = peek_range[1] - peek_range[0]
            h = base_h
            rect = (x, y, w, h)
            rects.append(rect)
        return rects

    def do(self, boundary2binimgs, line_rect, caffe_cls):
        boundaries, bin_images = [], []
        for boundary, bin_image in boundary2binimgs:
            boundaries.append(boundary)
            bin_images.append(bin_image)
        
        bin_image = bin_images[-1]
        self.rect_img_clf.caffe_cls = caffe_cls
        self.rect_img_clf.bin_image = None
        x, y, w, h = line_rect
        page_w = bin_image.shape[1]
        img_line = bin_image[y: y + h, x: x + w]
        char_w = page_w * self.char_set["width"]
        ocr_res = None
        ## first segmentation
        vertical_sum = np.sum(img_line, axis=0)
        peek_ranges = extract_peek_ranges_from_array(
            vertical_sum,
            minimun_val=10,
            minimun_range=2)

        rects = self.convert_peek_ranges_into_rects(
            peek_ranges, line_rect)
        self.rect_img_clf.char_set = self.char_set
        ocr_res = self.rect_img_clf.do_images_maxproba(
            rects, boundaries, bin_images)
        if ocr_res is not None:
            print("before merge..")
            print(ocr_res.encode("utf-8"))
            peek_ranges = merge_peek_ranges_mini_non_digits(
                peek_ranges, char_w, ocr_res)
            rects = self.convert_peek_ranges_into_rects(
                peek_ranges, line_rect)
            ocr_res = self.rect_img_clf.do_images_maxproba(
                rects, boundaries, bin_images)
            print("after merge...")
            print(ocr_res.encode("utf-8"))

#        ## end end segmenetation
#        if self.debug_path is not None:
#            path_debug_image_line = self.debug_path+"_line.jpg"
#            debug_img_line = np.copy(bin_image)
#            for rect in rects:
#                x = rect[0]
#                y = rect[1]
#                w = rect[2]
#                h = rect[3]
#                cv2.rectangle(debug_img_line,
#                              (x, y),
#                              (x + w, y + h),
#                              (255,255,255))
#            cv2.imwrite(path_debug_image_line, debug_img_line)
        return ocr_res