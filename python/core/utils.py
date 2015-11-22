#! /usr/bin/env python
'''
A utilities library for various io/data aggregation tasks
'''

import os
import re
from itertools import *
import subprocess
from subprocess import PIPE
import numpy as np
from pandas import DataFrame, Series
from core.image_scanner import ImageScanner
import PIL
import cv2
# ------------------------------------------------------------------------------

def get_report(y_true, y_pred):
    x = classification_report(y_true, y_pred)
    x = re.sub('avg / total', 'total', x)
    x = map(lambda x: re.split(' +', x), x.split('\n'))
    x = map(lambda x: filter(lambda x: x != '', x), x)
    x = filter(lambda x: x != [], x)
    report = DataFrame(x[1:])
    report.set_index(0, inplace=True)
    report.columns = x[0]
    return report
# ------------------------------------------------------------------------------

def pil_to_opencv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def opencv_to_pil(item):
    return PIL.Image.fromarray(item)

def generate_samples(image, y, params):
    scan = ImageScanner(image, **params)
    func = getattr(scan, params['scan_method'])
    return [[x, y, params] for x in func(**params)]

def get_channel_histogram(image, channel, bins=256, normalize=False, **kwargs):
    lut = {
        'r': 2, 'g': 1, 'b': 0,
        'h': 0, 's': 1, 'v': 2
          }
    output = cv2.calcHist([image],[lut[channel]], None, [bins], [0, 256])
    if normalize:
        output = cv2.normalize(output)
    return output.ravel()
# ------------------------------------------------------------------------------

def get_histograms(image, bins=256, normalize=False, color_space='rgb'):
    return {chan: get_channel_histogram(image, chan, bins=bins, normalize=normalize) for chan in color_space}

def generate_histograms(item, params, color_space='rgb'):
    img = opencv_to_pil(item)
    output = []
    for p in ImageScanner(img, **params).random_scan(params['patches']):
        patch = get_histograms(_pil_to_opencv(p), normalize=params['normalize'])
        output.append(patch)
    return output

def get_3d_histogram(image, bins=256, mask=None, normalize=False):
    hist = cv2.calcHist([image],[0,1,2], mask,
                        [np.sqrt(bins).astype(int)]*3,
                        [0, 256, 0, 256, 0, 256])
    if normalize:
        cv2.normalize(hist, hist)
    return hist
# ------------------------------------------------------------------------------

def plot_channel_histogram(image, channel, bins=256, normalize=False):
    lut = {
        'r': 'r', 'g': 'g', 'b': 'b',
        'h': 'w', 's': 'w', 'v': 'w'
          }
    hist = get_channel_histogram(image, channel, bins=bins, normalize=normalize)
    Series(hist).plot(color=lut[channel])

def plot_histograms(image, bins=256, normalize=False):
    for hist, color in get_histograms(image, bins=bins, normalize=normalize).iteritems():
        Series(hist).plot(color=color)

# ------------------------------------------------------------------------------
def execute_python_subshells(script, iterable):
    '''
    a simple hacky workaroud for multiprocessing's buginess
    executes a new python subshell per item

    Args:
        script (str): fullpath of python script to run (check /bin)
        iterable (iter): list of argument to provide each call
    
    Returns:
        None
    '''
    for item in iterable:
        cmd = script, ' '.join(item), '2>/dev/null &'
        cmd = ' '.join(cmd)
        subprocess.Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
# ------------------------------------------------------------------------------

__all__ = [
    'get_report',
    'pil_to_opencv',
    'opencv_to_pil',
    'generate_samples',
    'get_channel_histogram',
    'get_histograms',
    'generate_histograms',
    'get_3d_histogram',
    'plot_channel_histogram',
    'plot_histograms',
    'execute_python_subshells'
]

def main():
    pass

if __name__ == '__main__':
    help(main)