#! /usr/bin/env python

import os
import re
from itertools import *
from copy import copy
import multiprocessing
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.io.pytables import HDFStore
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import PIL
import cv2
from core.image_scanner import ImageScanner
from core.utils import *
# ------------------------------------------------------------------------------

# info utils
def get_info(source, spec=['name', 'extension'], sep=None, ignore=['\.DS_Store']):
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
    images = [source]
    if os.path.isdir(source):
        images = os.listdir(source)
    for regex in ignore:
        images = filter(lambda x: not re.search(regex, x), images)
    data = []
    for image in sorted(images):
        datum = []
        if sep:
            datum.extend(re.split(sep, image))
        else:
            temp = os.path.splitext(image)
            datum.append(temp[0])
            datum.append(temp[1].strip('.'))

        datum.append(os.path.join(source, image))
        data.append(datum)
    spec.append('source')
    return DataFrame(data, columns=spec)

def info_split(info, test_size=0.2):
    def _info_split(info, test_size=0.2):
        train_x, test_x, train_y, test_y = train_test_split(info, info.label, test_size=test_size)
        return DataFrame(train_x, columns=info.columns), DataFrame(test_x, columns=info.columns)
    
    train = []
    test = []
    for name in info.label.unique():
        x, y = _info_split(info[info.label == name], test_size=test_size)
        train.append(x)
        test.append(y)
    return pd.concat(train, axis=0), pd.concat(test, axis=0)
# ------------------------------------------------------------------------------

def process_data(info, features=['r', 'g', 'b', 'h', 's', 'v', 'fft_std', 'fft_max']):
    # create data from info
    data = info.copy()
    data.reset_index(drop=True, inplace=True)
    data = data[['source', 'label', 'params']]
    
    err = data.source.tolist()

    data.source = data.source.apply(lambda x: PIL.Image.open(x))
    data = data.apply(lambda x: 
        generate_samples(x['source'], x['label'], x['params']),
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
    fft = filter(lambda x: x in ['fft_std', 'fft_max'], features)
    
    # rgb distributions
    if rgb:
        temp = data[['bgr', 'params']].apply(lambda x: (x['bgr'], x['params']), axis=1)
        for chan in rgb:
            chan_data = temp.apply(lambda x: get_channel_histogram(x[0], chan, **x[1]))
            # data[chan] = chan_data.apply(lambda x: x.tolist())
            create_histogram_stats(data, chan_data, chan)

    # hsv distributions
    if hsv:
        try:
            data['hsv'] = data.bgr.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV))
        except:
            print(err)
            raise SyntaxError
        temp = data[['hsv', 'params']].apply(lambda x: (x['hsv'], x['params']), axis=1)
        for chan in hsv:
            chan_data = temp.apply(lambda x: get_channel_histogram(x[0], chan, **x[1]))
            # data[chan] = chan_data.apply(lambda x: x.tolist())
            create_histogram_stats(data, chan_data, chan)
    
        del data['hsv']
    
    # grain frequency
    if fft:
        data['gray'] = data.bgr.apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
        data.gray = data.gray.apply(lambda x: np.fft.hfft(x).astype(float))
        data.gray = data.gray.apply(lambda x: np.histogram(x.ravel(), bins=256)[0])
        if 'fft_std' in fft:
            data['fft_std'] = data.gray.apply(lambda x: x.std())
        if 'fft_max' in fft:
            data['fft_max'] = data.gray.apply(lambda x: x.max())

        del data['gray']
    
    del data['bgr']
    del data['params']
    
    # expand columns that contain lists
    # if rgb or hsv:
    #     data = _flatten(data)

    # shuffle data to destroy serial correlations
    index = data.index.tolist()
    np.random.shuffle(index)
    data = data.ix[index]
    data.reset_index(drop=True, inplace=True)

    # Normalize features
    cols = data.drop('y', axis=1).columns.tolist()
    ss = StandardScaler()
    data[cols] = ss.fit_transform(data[cols])
    
    return data

# multiproceesing
def _multi_get_data(args):
    return process_data(args[0], features=args[1])

def _batch_get_data(info, multiprocess=True, processes=24,
    features=['r', 'g', 'b', 'h', 's', 'v', 'fft_std', 'fft_max']):
    if not multiprocess:
        return process_data(info, features=features)

    pool = multiprocessing.Pool(processes=processes)
    iterable = [(row.to_frame().T, features) for i, row in info.iterrows()]
    data = pool.map(_multi_get_data, iterable)
    pool.terminate()

    # data = map(_multi_get_data, iterable) # for debugging

    data = pd.concat(data, axis=0)
    
    return data

def get_data(info, hdf_path, multiprocess=True, processes=24, write=True,
             features=['r', 'g', 'b', 'h', 's', 'v', 'fft_std', 'fft_max']):
    if not multiprocess:
        return process_data(info, features=features)
    
    # irregular index screws up index iterations
    info = info.copy()
    info.reset_index(drop=True, inplace=True)

    kwargs = {
        'features': features,
        'multiprocess': multiprocess,
        'processes': processes
    }

    batch_path = os.path.join('/var/tmp', 'hdf_batch')
    if os.path.exists(batch_path):
        os.remove(batch_path)
    os.mkdir(batch_path)

    n = info.shape[0]
    indices = range(0, n, processes)
    indices.append(n)
    indices = zip(indices, indices[1:])
    
    for i, (start, stop) in enumerate(indices):
        batch = info.ix[start:stop] #.copy()
        # batch.reset_index(drop=True, inplace=True)
        data = _batch_get_data(batch, **kwargs)
        
        filename = 'data.' + str(i).zfill(4) + '.hdf.batch'
        fullpath = os.path.join(batch_path, filename)
        hdf = HDFStore(fullpath)
        hdf['data'] = data
        hdf.close()
        
        print('indices {:>5} - {:<5} written to {:<5}'.format(start, stop, fullpath))
    
    batch = filter(lambda x: '.hdf.batch' in x, os.listdir(batch_path))
    batch = [os.path.join(batch_path, x) for x in batch]
    data = [pd.read_hdf(x, 'data') for x in batch]
    data = pd.concat(data, axis=0, ignore_index=True)
    
    # shuffle data to destroy serial correlations
    index = data.index.tolist()
    np.random.shuffle(index)
    data = data.ix[index]
    data.reset_index(drop=True, inplace=True)   
    
    if write:
        hdf = HDFStore(hdf_path)
        hdf['data'] = data
        hdf.close()

    return data
# ------------------------------------------------------------------------------

def compile_predictions(pred):
    data = DataFrame()
    data['yhat'] = pred
    data['confidence'] = 1.0
    data = data.groupby('yhat').agg(lambda x: x.sum() / data.shape[0])
    data.sort('confidence', ascending=False, inplace=True)
    data['label'] = data.index
    data.reset_index(drop=True, inplace=True)
    return data
# ------------------------------------------------------------------------------

def archive_data(train_info, test_info, hdf_path, cross_val=True,
                 multiprocess=True, processes=24,
                 features=['r', 'g', 'b', 'h', 's', 'v', 'fft_max', 'fft_std']):
    
    kwargs = {
        'features': features,
        'write': False,
        'multiprocess': multiprocess,
        'processes': processes
    }

    hdf = HDFStore(hdf_path)

    os.mkdir(hdf_path)
    batch = os.path.join(hdf_path, '.train')
    os.mkdir(batch)
    train = get_data(train_info, batch, **kwargs)
    train_x = train.drop('y', axis=1)
    train_y = train.y
    if cross_val:
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)
        hdf['train_x'] = DataFrame(train_x)
        hdf['valid_x'] = DataFrame(valid_x)
        hdf['train_y'] = Series(train_y)
        hdf['valid_y'] = Series(valid_y)
    else:
        hdf['train_x'] = train_x
        hdf['train_y'] = train_y

    batch = os.path.join(hdf_path, '.test')
    os.mkdir(batch)
    test = get_data(test_info, batch, **kwargs)
    test_x = test.drop('y', axis=1)
    test_y = test.y

    hdf['test_x'] = test_x
    hdf['test_y'] = test_y

    hdf.close()
    
    if cross_val:
        return train_x, valid_x, test_x, train_y, valid_y, test_y
    return train_x, test_x, train_y, test_y

def read_archive(hdf_path, items=['train_x', 'valid_x', 'test_x', 'train_y', 'valid_y', 'test_y']):
    hdf = HDFStore(hdf_path)
    output = map(lambda x: hdf[x], items)
    hdf.close()
    return output
# ------------------------------------------------------------------------------

__all__ = [
    'get_info',
    'info_split',
    'process_data',
    'get_data',
    'compile_predictions',
    'archive_data',
    'read_archive'
]

def main():
    pass

if __name__ == '__main__':
    help(main)