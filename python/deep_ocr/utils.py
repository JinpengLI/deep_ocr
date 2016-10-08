# -*- coding: utf-8 -*-

def trim_string(string_data):
    string_data = string_data.replace("    ", "")
    string_data = string_data.replace(" ", "")
    string_data = string_data.replace("\n", "")
    ### for string
    string_data = "".join(list(set(string_data)))
    return string_data
