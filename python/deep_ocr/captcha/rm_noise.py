# -*- coding: utf-8 -*-

import PIL.Image
import sys

# http://stackoverflow.com/questions/11253899/removing-the-background-noise-of-a-captcha-image-by-replicating-the-chopping-fil

class RMNoise(object):
    def __init__(self):
        pass

    def do_path(self, image_path, out_path, chop=2):
        # python chop.py [chop-factor] [in-file] [out-file]

        image = PIL.Image.open(image_path).convert('1')
        width, height = image.size
        data = image.load()

        # Iterate through the rows.
        for y in range(height):
            for x in range(width):
                # Make sure we're on a dark pixel.
                if data[x, y] > 128:
                    continue
                # Keep a total of non-white contiguous pixels.
                total = 0
                # Check a sequence ranging from x to image.width.
                for c in range(x, width):
                    # If the pixel is dark, add it to the total.
                    if data[c, y] < 128:
                        total += 1
                    # If the pixel is light, stop the sequence.
                    else:
                        break
                # If the total is less than the chop, replace everything with white.
                if total <= chop:
                    for c in range(total):
                        data[x + c, y] = 255
                # Skip this sequence we just altered.
                x += total

        # Iterate through the columns.
        for x in range(width):
            for y in range(height):
                # Make sure we're on a dark pixel.
                if data[x, y] > 128:
                    continue
                # Keep a total of non-white contiguous pixels.
                total = 0
                # Check a sequence ranging from y to image.height.
                for c in range(y, height):
                    # If the pixel is dark, add it to the total.
                    if data[x, c] < 128:
                        total += 1
                    # If the pixel is light, stop the sequence.
                    else:
                        break
                # If the total is less than the chop, replace everything with white.
                if total <= chop:
                    for c in range(total):
                        data[x, y + c] = 255
                # Skip this sequence we just altered.
                y += total

        image.save(out_path)