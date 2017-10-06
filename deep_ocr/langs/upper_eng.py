# -*- coding: utf-8 -*-

from deep_ocr.utils import trim_string

data = u'''
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
'''

data = trim_string(data)
