#! /usr/bin/env python
'''
A utilities library for various io/data aggregation tasks
'''
from __future__ import division, with_statement, print_function
from itertools import *
import os
import re
from collections import *
import subprocess
from subprocess import PIPE
import scipy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.metrics import classification_report
from core.image_scanner import ImageScanner
import PIL
import cv2
# from IPython import display
# ------------------------------------------------------------------------------

def get_report(y_true, y_pred):
    '''
    returns a classification report as a DataFrame, rather than as text

    Args:
        y_true (array-like):
            list of true labels

        y_pred (array-like):
            list of predicted labels

    Returns:
        classification report: DataFrame
    '''
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
    '''
    converts PIL.Image into cv2 image

    Args:
        image (PIL.Image):
            pillow image

    Returns:
        opencv object: cv2
        object is in BGR color space
    '''
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def opencv_to_pil(image):
    '''
    converts cv2 image into PIL.Image

    Args:
        image (cv2 image):
            cv2 image

    Returns:
        pillow image: PIL.Image
        object is in BGR color space
    '''
    return PIL.Image.fromarray(image)

def generate_samples(image, label, params):
    '''
    convenience function for  generating samples from a provided image along with its label and parameters

    Args:
        image (PIL.Image):
            pillow image

        label (str):
            image label

        params (dict):
            params to provide to ImageScanner

    Returns:
        matrix of patches: list
    '''
    scan = ImageScanner(image, **params)
    func = getattr(scan, params['scan_method'])
    return [[x, label, params] for x in func(**params)]

def get_channel_histogram(image, channel, bins=256, normalize=False, **kwargs):
    '''
    generates frequency data for a given channel of a provided image

    Args:
        image (cv2 image):
            opencv image to be processed

        channel (str):
            color channel to be processed
            acceptable values: r, g, b, h, s, v

        bins opt(int):
            number of bins to split histogram into
            default: 256 (number of channel values for sRGB images)

        normalize opt(bool):
            normalize histogram data
            default: False

    Returns:
        raveled array: numpy.array
    '''
    lut = {
        'r': 2, 'g': 1, 'b': 0,
        'h': 0, 's': 1, 'v': 2
    }
    output = cv2.calcHist([image],[lut[channel]], None, [bins], [0, 256])
    if normalize:
        output = cv2.normalize(output)
    return output.ravel()

def create_histogram_stats(data, chan_data, channel):
    '''
    convenience function for appending statics based upon provided histogram data to data

    Args:
        data (DataFrame): data to be appended to 

        chan_data (DataFrame): channel histogram data

        channel (str): name of channel

    Returns:
        None
    '''
    data[channel + '_' + 'mean']   = chan_data.apply(lambda x: x.mean() )
    data[channel + '_' + 'max']    = chan_data.apply(lambda x: x.max() )
    data[channel + '_' + 'argmax'] = chan_data.apply(lambda x: np.argmax(x) )
    data[channel + '_' + 'std']    = chan_data.apply(lambda x: x.std() )
    data[channel + '_' + 'skew']   = chan_data.apply(lambda x: scipy.stats.skew(x) )
    data[channel + '_' + 'kurt']   = chan_data.apply(lambda x: scipy.stats.kurtosis(x) )
# ------------------------------------------------------------------------------

def get_histograms(image, bins=256, normalize=False, colorspace='rgb'):
    '''
    generates histogram data for each channel of an image

    Args:
        image (cv2 image):
            opencv image to be processed

        bins opt(int):
            number of bins to split histogram into
            default: 256 (number of channel values for sRGB images)

        normalize opt(bool):
            normalize histogram data
            default: False

        colorspace opt(str):
            colorspace of provided image
            acceptable values: 'rgb', 'hsv'
            default: 'rgb'

    Returns:
        dict of channel histograms: dict
    '''
    return {chan: get_channel_histogram(image, chan, bins=bins, normalize=normalize) for chan in colorspace}
# ------------------------------------------------------------------------------

def plot_channel_histogram(image, channel, bins=256, normalize=False):
    '''
    plots a histogram of channel of a provided image

    Args:
        image (cv2 image):
            opencv image to be processed

        channel (str): color channel

        bins opt(int):
            number of bins to split histogram into
            default: 256 (number of channel values for sRGB images)

        normalize opt(bool):
            normalize histogram data
            default: False

    Returns:
        None
    '''
    lut = {
        'r': 'r', 'g': 'g', 'b': 'b',
        'h': 'w', 's': 'w', 'v': 'w'
          }
    hist = get_channel_histogram(image, channel, bins=bins, normalize=normalize)
    Series(hist).plot(color=lut[channel])

def plot_histograms(image, bins=256, normalize=False):
    '''
    plots a histogram of all channels of a provided image

    Args:
        image (cv2 image):
            opencv image to be processed

        bins opt(int):
            number of bins to split histogram into
            default: 256 (number of channel values for sRGB images)

        normalize opt(bool):
            normalize histogram data
            default: False

    Returns:
        None
    '''
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

# def show_image(response):
#     if not re.search('endgrain|database', response['snippet'], flags=re.IGNORECASE):
#         print(response['snippet'], response['displayLink'])
#         img = display.Image(url=r['link'], width=300, height=300)
#         display.display(img)
        
# def display_results(response):
#     for item in response:
#         print(item['snippet'], item['displayLink'])
#         img = display.Image(url=item['link'], width=300, height=300)
#         display.display(img)
# ------------------------------------------------------------------------------

__all__ = [
    'get_report',
    'pil_to_opencv',
    'opencv_to_pil',
    'generate_samples',
    'get_channel_histogram',
    'create_histogram_stats',
    'get_histograms',
    'plot_channel_histogram',
    'plot_histograms',
    'execute_python_subshells'
    # 'show_image',
    # 'display_results'
]

def main():
    pass

if __name__ == '__main__':
    help(main)