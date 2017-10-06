#!/usr/bin/env python

from __future__ import print_function

import sys
import glob
import os.path
from setuptools import setup

setup(
    name = 'deep_ocr',
    version = 'v0.1',
    author = "Jinpeng Li",
    description = "Make a Better Chinese Recognizer",
    packages = ["deep_ocr"],
    package_dir = {"deep_ocr": "deep_ocr"},
    )
