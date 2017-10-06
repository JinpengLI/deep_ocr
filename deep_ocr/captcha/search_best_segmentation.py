# -*- coding: utf-8 -*-

from deep_ocr.cv2_img_proc import PreprocessResizeKeepRatioFillBG
from deep_ocr.cv2_img_proc import PreprocessCropZeros

import numpy as np
import os
import cv2

class SearchBestSegmentation(object):

    def __init__(self, cls, cv2_grey_img, debug_path=None):
        self.cls = cls
        self.cv2_img = cv2_grey_img
        self.debug_path = debug_path

    def _extract_sub_img(self, cv2_img, rect):
        x, y, w, h = rect
        return cv2_img[y: y+h-1, x: x+w-1]

    def _extract_sub_imgs(self, cv2_img, segmentation):
        char_w = self.cls.width
        char_h = self.cls.height
        proc_resize = PreprocessResizeKeepRatioFillBG(
            width=char_w, height=char_h,
            auto_avoid_fill_bg=False,
            fill_bg=True, margin=2)
        crop_zeros = PreprocessCropZeros()
        sub_imgs = []
        for rect in segmentation:
            sub_img = self._extract_sub_img(cv2_img, rect)
            sub_img = crop_zeros.do(sub_img)
            sub_imgs.append(proc_resize.do(sub_img))
        return np.asarray(sub_imgs)/255.0

    def eval_segmentation(self, cv2_img, segmentation):
        sub_imgs = self._extract_sub_imgs(cv2_img, segmentation)
        tag_to_probas = self.cls.predict_cv2_imgs(sub_imgs)
        #compute the proba
        accumulate_proba = 1.0
        tags = []
        for tag_to_proba in tag_to_probas:
            tag = tag_to_proba[0][0]
            proba = tag_to_proba[0][1]
            accumulate_proba *= proba
            tags.append(tag)

        if self.debug_path is not None:
            import uuid
            sub_imgs_dir = os.path.join(self.debug_path, str(uuid.uuid1()))
            os.makedirs(sub_imgs_dir)
            for i, sub_img in enumerate(sub_imgs):
                image_path = os.path.join(sub_imgs_dir, "%d.jpg" % i)
                cv2.imwrite(image_path, sub_img*255.0)
            stat_path = os.path.join(sub_imgs_dir, "stat.txt")
            f_stat_path = open(stat_path, "w+")
            f_stat_path.write("".join(tags))
            f_stat_path.write("\n")
            f_stat_path.write("%f" % accumulate_proba)
            f_stat_path.write("\n")
            f_stat_path.close()
            sub_imgs_dir_pic = sub_imgs_dir + ".jpg"
            cv2_img_copy = np.copy(cv2_img)
            for one_segmentation in segmentation:
                left_x = one_segmentation[0]
                cv2.line(cv2_img_copy, (), )
        return accumulate_proba, tags


    def do(self, segmentations):
        eval_segmentations = []
        for segmentation in segmentations:
            accumulate_proba, tags = \
                self.eval_segmentation(self.cv2_img, segmentation)
#            print("=" * 60)
#            print("accumulate_proba=", accumulate_proba)
#            print("tags=", tags)
            eval_segmentations.append((tags, accumulate_proba))
        eval_segmentations = sorted(eval_segmentations, key=lambda x:x[1], reverse=True)
        return eval_segmentations
