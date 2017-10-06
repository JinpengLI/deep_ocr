# -*- coding: utf-8 -*-

import cv2
from deep_ocr.cv2_img_proc import PreprocessBackgroundMask
from deep_ocr.cv2_img_proc import PreprocessRemoveNonCharNoise

import numpy as np
from deep_ocr.utils import extract_peek_ranges_from_array
from deep_ocr.utils import merge_chars_into_line_segments
import os
import shutil
import sys, traceback

class Segmentation(object):

    def __init__(self, debug_path=None):
        self.debug_path = debug_path
        self.boundaries = [
                           ([0, 0, 0], [100, 100, 100]),
                           ([0, 0, 0], [150, 150, 150]),
                           ([0, 0, 0], [200, 200, 200]),
                          ]

    def check_if_good_boundary(self, boundary, norm_height, norm_width, color_img):
        preprocess_bg_mask = PreprocessBackgroundMask(boundary)
        char_w = norm_width / 20
        remove_noise = PreprocessRemoveNonCharNoise(char_w)

        id_card_img_mask = preprocess_bg_mask.do(color_img)
        id_card_img_mask[0:int(norm_height*0.05),:] = 0
        id_card_img_mask[int(norm_height*0.95): ,:] = 0
        id_card_img_mask[:, 0:int(norm_width*0.05)] = 0
        id_card_img_mask[:, int(norm_width*0.95):] = 0

        remove_noise.do(id_card_img_mask)

#        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#        mask = cv2.morphologyEx(id_card_img_mask, cv2.MORPH_CLOSE, se1)
#        id_card_img_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
#  
        ## remove right head profile
        left_half_id_card_img_mask = np.copy(id_card_img_mask)
        left_half_id_card_img_mask[:, norm_width/2:] = 0

        ## Try to find text lines and chars
        horizontal_sum = np.sum(left_half_id_card_img_mask, axis=1)
        line_ranges = extract_peek_ranges_from_array(horizontal_sum)

        return len(line_ranges) >= 5 and len(line_ranges) <= 7


    def do(self, color_img):

        shape = color_img.shape

        norm_height = shape[0]
        norm_width = shape[1]

        gray_id_card_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_id_card_img = clahe.apply(gray_id_card_img)

        gray_id_card_img = 255 - gray_id_card_img

        best_boundary = None
        for boundary in self.boundaries:
            if self.check_if_good_boundary(
                    boundary,
                    norm_height, norm_width,
                    color_img):
                best_boundary = boundary
                break
        if best_boundary is None:
            return {}

        boundary = best_boundary
        ## boundary = ([0, 0, 0], [100, 100, 100])
        preprocess_bg_mask = PreprocessBackgroundMask(boundary)
        id_card_img_mask = preprocess_bg_mask.do(color_img)
        id_card_img_mask[0:int(norm_height*0.05),:] = 0
        id_card_img_mask[int(norm_height*0.95): ,:] = 0
        id_card_img_mask[:, 0:int(norm_width*0.05)] = 0
        id_card_img_mask[:, int(norm_width*0.95):] = 0

#        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#        mask = cv2.morphologyEx(id_card_img_mask, cv2.MORPH_CLOSE, se1)
#        id_card_img_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

        ## remove right head profile
        left_half_id_card_img_mask = np.copy(id_card_img_mask)
        left_half_id_card_img_mask[:, norm_width/2:] = 0

        ## Try to find text lines and chars
        vertical_peek_ranges2d = []
        horizontal_sum = np.sum(left_half_id_card_img_mask, axis=1)
        line_ranges = extract_peek_ranges_from_array(horizontal_sum)

        ## char extraction
        for line_range in line_ranges:
            start_y, end_y = line_range
            end_y += 1
            line_img = id_card_img_mask[start_y: end_y]
            vertical_sum = np.sum(line_img, axis=0)
            vertical_peek_ranges = extract_peek_ranges_from_array(
                vertical_sum,
                minimun_val=40,
                minimun_range=1)
            vertical_peek_ranges2d.append(vertical_peek_ranges)
        vertical_peek_ranges2d = merge_chars_into_line_segments(vertical_peek_ranges2d)
        img_gray_texts = cv2.bitwise_and(gray_id_card_img,
                                         gray_id_card_img,
                                         mask=id_card_img_mask)

        key_to_segmentation = {}
        try:
            ## name extraction
            range_y = line_ranges[0]
            range_x = vertical_peek_ranges2d[0][0]
            start_x, end_x = range_x
            start_y, end_y = range_y
            w = end_x - start_x
            h = end_y - start_y
            key_to_segmentation["name"] = [(start_x, start_y, w, h), ]
            ## sex extraction
            range_y = line_ranges[1]
            range_x = vertical_peek_ranges2d[1][0]
            start_x, end_x = range_x
            start_y, end_y = range_y
            w = end_x - start_x
            h = end_y - start_y
            key_to_segmentation["sex"] = [(start_x, start_y, w, h), ]
            ## minzu extraction
            range_y = line_ranges[1]
            range_x = vertical_peek_ranges2d[1][1]
            start_x, end_x = range_x
            start_y, end_y = range_y
            w = end_x - start_x
            h = end_y - start_y
            key_to_segmentation["minzu"] = [(start_x, start_y, w, h), ]
            ## year extraction
            range_y = line_ranges[2]
            range_x = vertical_peek_ranges2d[2][0]
            start_x, end_x = range_x
            start_y, end_y = range_y
            w = end_x - start_x
            h = end_y - start_y
            key_to_segmentation["year"] = [(start_x, start_y, w, h), ]
            ## month extraction
            range_y = line_ranges[2]
            range_x = vertical_peek_ranges2d[2][1]
            start_x, end_x = range_x
            start_y, end_y = range_y
            w = end_x - start_x
            h = end_y - start_y
            key_to_segmentation["month"] = [(start_x, start_y, w, h), ]
            ## day extraction
            range_y = line_ranges[2]
            range_x = vertical_peek_ranges2d[2][2]
            start_x, end_x = range_x
            start_y, end_y = range_y
            w = end_x - start_x
            h = end_y - start_y
            key_to_segmentation["day"] = [(start_x, start_y, w, h), ]
            ## address extraction
            key_to_segmentation["address"] = []
            first_line = line_ranges[3][0]
            first_line_range_x = vertical_peek_ranges2d[3][0]
            first_line_start_x = first_line_range_x[0]
            first_line_w = first_line_range_x[1] - first_line_start_x
            for i, line_range in enumerate(line_ranges):
                if i >= 3:
                    range_y = line_range
                    range_x = vertical_peek_ranges2d[i][0]
                    start_x, end_x = range_x
                    start_y, end_y = range_y
                    if abs(first_line_start_x - start_x)> int(first_line_w * 0.05):
                        break
                    w = end_x - start_x
                    h = end_y - start_y
                    key_to_segmentation["address"].append((start_x, start_y, w, h))
            ## id extraction
            range_y = line_ranges[-1]
            range_x = vertical_peek_ranges2d[-1][0]
            start_x, end_x = range_x
            start_y, end_y = range_y
            w = end_x - start_x
            h = end_y - start_y
            key_to_segmentation["id"] = [(start_x, start_y, w, h), ]
        except:
            print "Exception in user code:"
            print '-' * 60
            traceback.print_exc(file=sys.stdout)
            print '-' * 60
            key_to_segmentation = {}

        debug_path = self.debug_path
        if debug_path is not None:
            import random

            if os.path.isdir(debug_path):
                shutil.rmtree(debug_path)
            os.makedirs(debug_path)

            debug_image_path = os.path.join(debug_path, "01_origin_img.jpg")
            debug_gray_image_path = os.path.join(debug_path, "01_gray_img.jpg")
            debug_image_mask_path = os.path.join(debug_path, "02_mask.jpg")
            debug_image_mask_text_lines_path = os.path.join(debug_path, "03_mask_text_lines.jpg")
            debug_image_left_mask_path = os.path.join(debug_path, "04_left_mask.jpg")
            debug_image_gray_texts_path = os.path.join(debug_path, "05_gray_texts.jpg")
            debug_image_chars_path = os.path.join(debug_path, "06_origin_img_chars.jpg")
            debug_image_key_to_segments_path = os.path.join(debug_path, "07_origin_img_key_to_segments.jpg")

            cv2.imwrite(debug_image_path, color_img)
            cv2.imwrite(debug_gray_image_path, 255 - gray_id_card_img)
            id_card_img_chars = np.copy(color_img)
            cv2.imwrite(debug_image_mask_path, id_card_img_mask)
            id_card_img_mask_text_lines = np.copy(id_card_img_mask)
    
            for i, line_range in enumerate(line_ranges):
                start_y, end_y = line_range
                id_card_img_mask_text_lines[start_y, :] = 255
                id_card_img_mask_text_lines[end_y, :] = 255
    
            color = (255, 0, 0)
            for i, line_range in enumerate(line_ranges):
                start_y, end_y = line_range
                for vertical_peek_range in vertical_peek_ranges2d[i]:
                    start_x, end_x = vertical_peek_range
                    cv2.rectangle(id_card_img_chars,
                                  (start_x, start_y),
                                  (end_x+1, end_y+1),
                                  color)

            key_to_segments_img = np.copy(color_img)
            for key in key_to_segmentation:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                rects = key_to_segmentation[key]
                for rect in rects:
                    pt1 = (rect[0], rect[1])
                    pt2 = (rect[0] + rect[2], rect[1] + rect[3])
                    cv2.rectangle(key_to_segments_img, pt1, pt2, color)
            
            cv2.imwrite(debug_image_mask_text_lines_path, id_card_img_mask_text_lines)
            
            cv2.imwrite(debug_image_left_mask_path, left_half_id_card_img_mask)
            cv2.imwrite(debug_image_gray_texts_path, img_gray_texts)
            cv2.imwrite(debug_image_chars_path, id_card_img_chars)
            cv2.imwrite(debug_image_key_to_segments_path, key_to_segments_img)
        return key_to_segmentation
            
