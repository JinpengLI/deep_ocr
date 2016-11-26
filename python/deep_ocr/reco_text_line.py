# -*- coding: utf-8 -*-

import numpy as np
import cv2
from deep_ocr.cv2_img_proc import PreprocessCropZeros
from deep_ocr.cv2_img_proc import PreprocessResizeKeepRatioFillBG
from deep_ocr.utils import extract_peek_ranges_from_array
from deep_ocr.utils import merge_peek_ranges

class RectImageClassifier(object):
    def __init__(self, caffe_cls, image, char_set,
                 caffe_cls_width=64, caffe_cls_height=64):
        self.caffe_cls = caffe_cls
        self.image = image
        self.cache_res = {}
        self.char_set = char_set
        self.caffe_cls_width = caffe_cls_width
        self.caffe_cls_height = caffe_cls_height

    def do(self, rects, boundary):
        rects_to_reco = []
        for rect in rects:
            key = (rect, boundary)
            if key not in self.cache_res:
                rects_to_reco.append(rect)
        image = self.image
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
                        self.cache_res[key] = char_p[0]
                        break
            else:
                self.cache_res[key] = item[0][0]
        ocr_res = ""
        for rect in rects:
            key = (rect, boundary)
            ocr_res += self.cache_res[key]
        return ocr_res



class RecoTextLine(object):
    def __init__(self, caffe_cls,
                 caffe_cls_width,
                 caffe_cls_height,
                 page_w=500,
                 char_set=set(),
                 debug_path=None,
                 boundary=(0,0,0)):
        self.caffe_cls = caffe_cls
        self.char_set = char_set
        self.caffe_cls_width = caffe_cls_width
        self.caffe_cls_height = caffe_cls_height
        self.debug_path = debug_path
        self.page_w = page_w
        self.boundary = boundary

    def do(self, img_line):
        rect_img_clf = RectImageClassifier(
            self.caffe_cls,
            img_line,
            self.char_set,
            caffe_cls_width=self.caffe_cls_width,
            caffe_cls_height=self.caffe_cls_height)

        height = img_line.shape[0]
        vertical_sum = np.sum(img_line, axis=0)
        char_w = self.page_w * self.char_set["width"]
        ## first segmentation
        peek_ranges = extract_peek_ranges_from_array(
            vertical_sum,
            minimun_val=10,
            minimun_range=2)

        peek_ranges = merge_peek_ranges(peek_ranges, char_w)
        rects = []
        for peek_range in peek_ranges:
            x = peek_range[0]
            y = 0
            w = peek_range[1] - x
            h = height
            rect = (x, y, w, h)
            rects.append(rect)
        ocr_res = rect_img_clf.do(rects, self.boundary)

        ## end end segmenetation
        print(ocr_res.encode("utf-8"))
        if self.debug_path is not None:
            path_debug_image_line = self.debug_path+"_line.jpg"
            debug_img_line = np.copy(img_line)
            for peek_range in peek_ranges:
                x = peek_range[0]
                y = 0
                w = peek_range[1] - x
                h = height
                cv2.rectangle(debug_img_line,
                              (x, y),
                              (x+w+1, y+h+1),
                              (255,255,255))
            cv2.imwrite(path_debug_image_line, debug_img_line)
        return ocr_res