# -*- coding: utf-8 -*-

import numpy as np

def trim_string(string_data):
    string_data = string_data.replace("    ", "")
    string_data = string_data.replace(" ", "")
    string_data = string_data.replace("\n", "")
    ### for string
    string_data = "".join(list(set(string_data)))
    return string_data


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

def merge_chars_into_line_segments(ranges2d):
    for i, ranges in enumerate(ranges2d):
        m_w = compute_median_w_from_ranges(ranges)
        new_ranges = []
        for j, range_pair in enumerate(ranges):
            if len(new_ranges) == 0:
                new_ranges.append(range_pair)
            else:
                start_x, end_x = range_pair
                pre_start_x, pre_end_x = new_ranges[-1]
                if start_x > pre_start_x:
                    if start_x - pre_end_x < m_w *2:
                        new_ranges[-1] = (pre_start_x, end_x)
                    else:
                        new_ranges.append(range_pair)
        ranges2d[i] = new_ranges
    return ranges2d
