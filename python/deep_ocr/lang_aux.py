# -*- coding: utf-8 -*-


from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import traceback
import sys
import os
import numpy as np
import cv2

from deep_ocr.utils import trim_string
from deep_ocr.cv2_img_proc import FindImageBBox
from deep_ocr.cv2_img_proc import PreprocessResizeKeepRatioFillBG


class LangCharsGenerate(object):
    def __init__(self, langs):
        self.langs = langs

    def do(self, ):
        lang_list = self.langs.split("+")
        lang_chars = ""
        for lang in lang_list:
            lang_module = "deep_ocr.langs.%s" % lang
            lang_module_data = __import__(lang_module, fromlist=[''])
            lang_chars += lang_module_data.data
        trim_string(lang_chars)
        return lang_chars


class FontCheck(object):

    def __init__(self, lang_chars, width=32, height=32):
        self.lang_chars = lang_chars
        self.width = width
        self.height = height

    def do(self, font_path):
        width = self.width
        height = self.height
        try:
            for i, char in enumerate(self.lang_chars):
                img = Image.new("RGB", (width, height), "black")
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype(font_path, int(width * 0.9),)
                draw.text((0, 0), char, (255, 255, 255),
                          font=font)
                data = list(img.getdata())
                sum_val = 0
                for i_data in data:
                    sum_val += sum(i_data)
                if sum_val < 2:
                    return False
        except:
            print("fail to load:%s" % font_path)
            traceback.print_exc(file=sys.stdout)
            return False
        return True


class Font2Image(object):

    def __init__(self,
                 width, height,
                 need_crop, margin):
        self.width = width
        self.height = height
        self.need_crop = need_crop
        self.margin = margin

    def do(self, font_path, char, path_img):
        find_image_bbox = FindImageBBox()
        img = Image.new("RGB", (self.width, self.height), "black")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(self.width * 0.7),)
        draw.text((0, 0), char, (255, 255, 255),
                  font=font)
        data = list(img.getdata())
        sum_val = 0
        for i_data in data:
            sum_val += sum(i_data)
        if sum_val > 2:
            np_img = np.asarray(data, dtype='uint8')
            np_img = np_img[:, 0]
            np_img = np_img.reshape((self.height, self.width))
            cropped_box = find_image_bbox.do(np_img)
            left, upper, right, lower = cropped_box
            np_img = np_img[upper: lower + 1, left: right + 1]
            if not self.need_crop:
                preprocess_resize_keep_ratio_fill_bg = \
                    PreprocessResizeKeepRatioFillBG(self.width, self.height,
                                                    fill_bg=False,
                                                    margin=self.margin)
                np_img = preprocess_resize_keep_ratio_fill_bg.do(
                    np_img)
            cv2.imwrite(path_img, np_img)
        else:
            print("%s doesn't exist." % path_img)


if __name__ == "__main__":
    lang_chars_gen = LangCharsGenerate("digits+eng")
    lang_chars = lang_chars_gen.do()
    font_check = FontCheck(lang_chars)

    font_dir = "/root/workspace/deep_ocr_fonts/chinese_fonts/"
    for font_name in os.listdir(font_dir):
        font_path = os.path.join(font_dir, font_name)
        print("font_path:", font_path)
        lang_chars_gen = LangCharsGenerate("chi_sim")
        lang_chars = lang_chars_gen.do()
        print("char len=", len(lang_chars))
        #print(lang_chars.encode("utf-8"))
        font_check = FontCheck(lang_chars)
        print("can cover all the chars?:", font_check.do(font_path))
    
