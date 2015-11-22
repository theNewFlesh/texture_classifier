#! /usr/bin/env python

import os
import re
from itertools import *
from copy import copy
import multiprocessing
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.io.pytables import HDFStore
import PIL
import cv2
from core.image_scanner import ImageScanner
from core.utils import *
# ------------------------------------------------------------------------------

# info utils
def get_info(source, spec, ignore=['\.DS_Store']):
    '''
    creates a descriptive DataFrame based upon files contained with a source directory

    Args:
        source (str): fullpath to directory of files

        spec (list): naming specification of files (dotslot syntax)

        ignore (Optional(list)):
            list of regex patterns used for ignoring files
            default: ['\.DS_Store']

    Returns:
        DataFrame: an info DataFrame
    '''
    spec = copy(spec)
    images = os.listdir(source)
    for regex in ignore:
        images = filter(lambda x: not re.search(regex, x), images)
    data = []
    for image in sorted(images):
        datum = []
        datum.extend(image.split('.'))
        datum.append(os.path.join(source, image))
        data.append(datum)
    spec.append('source')
    return DataFrame(data, columns=spec)

def info_split(info, test_size=0.2):
    def _info_split(info, test_size=0.2):
        train_x, test_x, train_y, test_y = train_test_split(info, info.common_name, test_size=test_size)
        return DataFrame(train_x, columns=info.columns), DataFrame(test_x, columns=info.columns)
    
    train = []
    test = []
    for name in info.common_name.unique():
        x, y = _info_split(info[info.common_name == name], test_size=test_size)
        train.append(x)
        test.append(y)
    return pd.concat(train, axis=0), pd.concat(test, axis=0)
# ------------------------------------------------------------------------------

def _get_data(info, features=['r', 'g', 'b', 'h', 's', 'v', 'fft_var', 'fft_max']):
    # create data from info
    data = info.copy()
    data = data[['source', 'common_name', 'params']]
    data.source = data.source.apply(lambda x: PIL.Image.open(x))
    data = data.apply(lambda x: 
        generate_samples(x['source'], x['common_name'], x['params']),
        axis=1
    )
    # create new expanded dataframe
    data = list(chain(*data.tolist()))
    data = DataFrame(data, columns=['x', 'y', 'params'])
    data['bgr'] = data.x.apply(pil_to_opencv)
    
    del data['x']
    
    # create feature lists
    rgb = filter(lambda x: x in list('rgb'), features)
    hsv = filter(lambda x: x in list('hsv'), features)
    fft = filter(lambda x: x in ['fft_var', 'fft_max'], features)
    
    # rgb distributions
    if rgb:
        temp = data[['bgr', 'params']].apply(lambda x: (x['bgr'], x['params']), axis=1)
        for chan in rgb:
            c = temp.apply(lambda x: get_channel_histogram(x[0], chan, **x[1]))
            data[chan] = c.apply(lambda x: x.tolist())

    # hsv distributions
    if hsv:
        data['hsv'] = data.bgr.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV))
        temp = data[['hsv', 'params']].apply(lambda x: (x['hsv'], x['params']), axis=1)
        for chan in hsv:
            c = temp.apply(lambda x: get_channel_histogram(x[0], chan, **x[1]))
            data[chan] = c.apply(lambda x: x.tolist())
    
        del data['hsv']
    
    # grain frequency
    if fft:
        data['gray'] = data.bgr.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
        data.gray = data.gray.apply(lambda x: np.fft.hfft(x).astype(float))
        data.gray = data.gray.apply(lambda x: np.histogram(x.ravel(), bins=256)[0])
        data.gray = data.gray.apply(lambda x: StandardScaler().fit_transform(x))
        if 'fft_var' in fft:
            data['fft_var'] = data.gray.apply(lambda x: x.var())
        if 'fft_max' in fft:
            data['fft_max'] = data.gray.apply(lambda x: x.max())

        del data['gray']
    
    del data['bgr']
    del data['params']
    
    # expand columns that contain lists
    if rgb or hsv:
        sdf = SparseDataFrame(data)
        data = sdf.flatten(dtype=list)

    # shuffle data to destroy serial correlations
    index = data.index.tolist()
    np.random.shuffle(index)
    data = data.ix[index]
    data.reset_index(drop=True, inplace=True)
    
    return data

# multiproceesing
def _multi_get_data(args):
    return get_data(args[0], features=args[1])

def get_data(info, features=['r', 'g', 'b', 'h', 's', 'v', 'fft_var', 'fft_max'],
             multiprocess=True, processes=24):
    if not multiprocess:
        return _get_data(info features=features)

    pool = multiprocessing.Pool(processes=processes)
    iterable = [(row.to_frame().T, features) for i, row in info.iterrows()]
    data = pool.map(_multi_get_data, iterable)
    pool.close()
    data = pd.concat(data, axis=0)

    # shuffle data to destroy serial correlations
    index = data.index.tolist()
    np.random.shuffle(index)
    data = data.ix[index]
    data.reset_index(drop=True, inplace=True)
    
    return data
# ------------------------------------------------------------------------------

def archive_data(train, test, hdf_path=None):
    hdf = {}
    if hdf_path:
        hdf = HDFStore(hdf_path)

    train = get_data(train)
    train_x, valid_x, train_y, valid_y = train_test_split(train.drop('y', axis=1), train.y, test_size=0.2)
    hdf['train_x'] = train_x
    hdf['valid_x'] = valid_x
    hdf['train_y'] = train_y
    hdf['valid_y'] = valid_y

    test = get_data(test)
    test_x = test.drop('y', axis=1)
    test_y = test.y

    hdf['test_x'] = test_x
    hdf['test_y'] = test_y

    if write_hdf:
        hdf.close()
    
    return train_x, valid_x, train_y, valid_y, test_x, test_y
# ------------------------------------------------------------------------------

__all__ = [
    'get_info',
    'info_split',
    'pil_to_opencv',
    'opencv_to_pil',
    'generate_samples',
    'get_channel_histogram',
    'get_data',
    'archive_data',
    'get_report'
]

def main():
    pass

if __name__ == '__main__':
    help(main)