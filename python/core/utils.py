#! /usr/bin/env python
'''
A utilities library for various io/data aggregation tasks
'''

import os
import re
from itertools import *
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

def create_histogram_stats(data, chan_data, channel):
    data[channel + '_' + 'mean']   = chan_data.apply(lambda x: x.mean() )
    data[channel + '_' + 'max']    = chan_data.apply(lambda x: x.max() )
    data[channel + '_' + 'argmax'] = chan_data.apply(lambda x: np.argmax(x) )
    data[channel + '_' + 'std']    = chan_data.apply(lambda x: x.std() )
    data[channel + '_' + 'skew']   = chan_data.apply(lambda x: scipy.stats.skew(x) )
    data[channel + '_' + 'kurt']   = chan_data.apply(lambda x: scipy.stats.kurtosis(x) )
# ------------------------------------------------------------------------------

def get_histograms(image, bins=256, normalize=False, colorspace='rgb'):
    return {chan: get_channel_histogram(image, chan, bins=bins, normalize=normalize) for chan in colorspace}

def generate_histograms(item, params, colorspace='rgb'):
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

def _flatten(data, columns=None, prefix=True, drop=True):
        '''Split items of iterable elements into separate columns (ripped from sparse)

        Args:
            dtype (type, optional): Columns types to be split. Default: dict
            prefix (bool, optional): Append original column name as a prefix to new columns

        Returns: 
            Flattened DataFrame

        Example:
            >>> print sdf.data
                               foo             bar
            0  {u'a': 1, u'b': 10}     some string
            1  {u'a': 2, u'b': 20}  another string
            2  {u'a': 3, u'b': 30}            blah

            >>> sdf.flatten(inplace=True)
            >>> print sdf.data
                foo_a    foo_b             bar
            0       1       10     some string
            1       2       20  another string
            2       3       30            blah
        '''
        def _reorder_columns(columns, index):
            new_cols = []
            for col in columns:
                if col in index:
                    if not drop:
                        new_cols.append(col)
                    new_cols.extend( index[col] )
                else:
                    new_cols.append(col)
            return new_cols

        col_index = OrderedDict()
        def _flatten(data, columns):
            for col in columns:
                col_index[col] = [] 
            frames = []
            for col in columns:
                frame = DataFrame(data[col].tolist())
                if prefix:
                    columns = {}
                    for k in frame.columns:
                        columns[k] = str(col) + '_' + str(k)
                    frame.rename(columns=columns, inplace=True)
                frames.append(frame)
                col_index[col].extend( frame.columns.tolist() )
            data = pd.concat(frames, axis=1)
            return data
        
        flatdata = data
        old_cols = data.columns.tolist()

        # determine flatenable columns via column mask
        if columns:
            flatdata = flatdata[columns]
        else:
            mask = data.applymap(lambda x: isinstance(x, list))
            iterables = data[mask]
            iterables = iterables.dropna(how='all', axis=1)
            columns = iterables.columns.tolist()
        
        # Get right-hand flattened columns
        flatdata = _flatten(flatdata, columns)
        
        old_cols = data.columns.tolist()

        # drop original columns
        if drop:
            data = data.T.drop(columns).T

        # attach right-hand flattened columns to  original columns
        data = pd.concat([data, flatdata], axis=1)

        return data
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
    'generate_histograms',
    'get_3d_histogram',
    'plot_channel_histogram',
    'plot_histograms',
    '_flatten',
    'execute_python_subshells'
    # 'show_image',
    # 'display_results'
]

def main():
    pass

if __name__ == '__main__':
    help(main)