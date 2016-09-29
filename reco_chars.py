# -*- coding: utf-8 -*-
import caffe
import json
import numpy as np
import os
import cv2
import shutil
import copy

class CaffeCls(object):
    def __init__(self, 
                 model_def,
                 model_weights,
                 y_tag_json_path,
                 is_mode_cpu=True,
                 width=64,
                 height=64):
        self.net = caffe.Net(model_def,
            model_weights,
            caffe.TEST)
        if is_mode_cpu:
            caffe.set_mode_cpu()
        self.y_tag_json = json.load(open(y_tag_json_path, "r"))
        self.width = width
        self.height = height

    def predict_cv2_img(self, cv2_img):
        shape = cv2_img.shape
        cv2_imgs = cv2_img.reshape((1, shape[0], shape[1]))
        return self.predict_cv2_imgs(cv2_imgs)[0]


    def _predict_cv2_imgs_sub(self, cv2_imgs, pos_start, pos_end):
        cv2_imgs_sub = cv2_imgs[pos_start: pos_end]

        self.net.blobs['data'].reshape(cv2_imgs_sub.shape[0], 1,
                                       self.width, self.height)
        self.net.blobs['data'].data[...] = cv2_imgs_sub.reshape(
            (cv2_imgs_sub.shape[0], 1, self.width, self.height))
        output = self.net.forward()

        output_tag_to_max_proba = []

        num_sample = cv2_imgs_sub.shape[0]
        for i in range(num_sample):
            output_prob = output['prob'][i]
            output_prob_index = sorted(
                range(len(output_prob)),
                key=lambda x:output_prob[x],
                reverse=True)            
            output_tag_to_probas = []
            for index in output_prob_index:
                item = (self.y_tag_json[str(index)],
                        output_prob[index])
                output_tag_to_probas.append(item)
            # output_tag_to_probas = output_tag_to_probas[:2]
            output_tag_to_max_proba.append(output_tag_to_probas)
        return output_tag_to_max_proba

    def predict_cv2_imgs(self, cv2_imgs, step=50):
        output_tag_to_max_proba = []
        num_sample = cv2_imgs.shape[0]
        for i in range(0, num_sample, step):
            pos_end = min(num_sample, (i + step))
            output_tag_to_max_proba += \
                self._predict_cv2_imgs_sub(cv2_imgs, i, pos_end)
        return output_tag_to_max_proba

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

    def __init__(self, width, height, fill_bg=False,
                 auto_avoid_fill_bg=True, margin=None):
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

def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def compute_median_w_from_ranges(peek_ranges):
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    return median_w

def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i]/median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges


if __name__ == "__main__":

    norm_width = 64
    norm_height = 64

    base_dir = "/workspace/data/chongdata_caffe_cn_sim_digits_64_64"
    model_def = os.path.join(base_dir, "deploy_lenet_train_test.prototxt")
    model_weights = os.path.join(base_dir, "lenet_iter_50000.caffemodel")
    y_tag_json_path = os.path.join(base_dir, "y_tag.json")
    caffe_cls = CaffeCls(model_def, model_weights, y_tag_json_path)

    test_image = "/opt/deep_ocr/test_data.png"

    debug_dir = "/tmp/debug_dir"
    if debug_dir is not None:
        if os.path.isdir(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)

    cv2_color_img = cv2.imread(test_image)
    
    resize_keep_ratio = PreprocessResizeKeepRatio(1024, 1024)
    cv2_color_img = resize_keep_ratio.do(cv2_color_img)    

    cv2_img = cv2.cvtColor(cv2_color_img, cv2.COLOR_RGB2GRAY)
    height, width = cv2_img.shape

    adaptive_threshold = cv2.adaptiveThreshold(
        cv2_img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY, 11, 2)
    adaptive_threshold = 255 - adaptive_threshold

    ## Try to find text lines and chars
    horizontal_sum = np.sum(adaptive_threshold, axis=1)
    peek_ranges = extract_peek_ranges_from_array(horizontal_sum)
    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek_ranges_from_array(
            vertical_sum,
            minimun_val=40,
            minimun_range=1)
        vertical_peek_ranges = median_split_ranges(vertical_peek_ranges)
        vertical_peek_ranges2d.append(vertical_peek_ranges)

    ## remove noise such as comma
    filtered_vertical_peek_ranges2d = []
    for i, peek_range in enumerate(peek_ranges):
        new_peek_range = []
        median_w = compute_median_w_from_ranges(vertical_peek_ranges2d[i])
        for vertical_range in vertical_peek_ranges2d[i]:
            if vertical_range[1] - vertical_range[0] > median_w*0.7:
                new_peek_range.append(vertical_range)
        filtered_vertical_peek_ranges2d.append(new_peek_range)
    vertical_peek_ranges2d = filtered_vertical_peek_ranges2d


    char_imgs = []
    crop_zeros = PreprocessCropZeros()
    resize_keep_ratio = PreprocessResizeKeepRatioFillBG(
        norm_width, norm_height, fill_bg=False, margin=4)
    for i, peek_range in enumerate(peek_ranges):
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            w = vertical_range[1] - x
            h = peek_range[1] - y
            char_img = adaptive_threshold[y:y+h+1, x:x+w+1]
            char_img = crop_zeros.do(char_img)
            char_img = resize_keep_ratio.do(char_img)
            char_imgs.append(char_img)

    np_char_imgs = np.asarray(char_imgs)

    output_tag_to_max_proba = caffe_cls.predict_cv2_imgs(np_char_imgs)

    ocr_res = ""
    for item in output_tag_to_max_proba:
        ocr_res += item[0][0]
    print(ocr_res.encode("utf-8"))

    if debug_dir is not None:
        path_adaptive_threshold = os.path.join(debug_dir,
                                               "adaptive_threshold.jpg")
        cv2.imwrite(path_adaptive_threshold, adaptive_threshold)
        seg_adaptive_threshold = cv2_color_img

#        color = (255, 0, 0)
#        for rect in rects:
#            x, y, w, h = rect
#            pt1 = (x, y)
#            pt2 = (x + w, y + h)
#            cv2.rectangle(seg_adaptive_threshold, pt1, pt2, color)

        color = (0, 255, 0)
        for i, peek_range in enumerate(peek_ranges):
            for vertical_range in vertical_peek_ranges2d[i]:
                x = vertical_range[0]
                y = peek_range[0]
                w = vertical_range[1] - x
                h = peek_range[1] - y
                pt1 = (x, y)
                pt2 = (x + w, y + h)
                cv2.rectangle(seg_adaptive_threshold, pt1, pt2, color)
            
        path_seg_adaptive_threshold = os.path.join(debug_dir,
                                                   "seg_adaptive_threshold.jpg")
        cv2.imwrite(path_seg_adaptive_threshold, seg_adaptive_threshold)

        debug_dir_chars = os.path.join(debug_dir, "chars")
        os.makedirs(debug_dir_chars)
        for i, char_img in enumerate(char_imgs):
            path_char = os.path.join(debug_dir_chars, "%d.jpg" % i)
            cv2.imwrite(path_char, char_img)
            
            
