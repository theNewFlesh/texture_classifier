'''
A utilities library for various io/data aggregation tasks
'''

import os
import re
from itertools import *
from copy import copy
import subprocess
from subprocess import PIPE
import multiprocessing
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.io.pytables import HDFStore
from PIL import Image
from core.image_scanner import ImageScanner
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

def get_info(root, spec, ignore=['\.DS_Store']):
    '''
    creates a descriptive DataFrame based upon files contained with a root directory

    Args:
        root (str): fullpath to directory of files

        spec (list): naming specification of files (dotslot syntax)

        ignore (Optional(list)):
            list of regex patterns used for ignoring files
            default: ['\.DS_Store']

    Returns:
        DataFrame: an info DataFrame
    '''
    spec = copy(spec)
    images = os.listdir(root)
    for regex in ignore:
        images = filter(lambda x: not re.search(regex, x), images)
    data = []
    for image in sorted(images):
        datum = []
        datum.extend(image.split('.'))
        datum.append(os.path.join(root, image))
        data.append(datum)
    spec.append('fullpath')
    return DataFrame(data, columns=spec)

def get_data(fullpath, params):
    '''
    creates raveled data based upon an image and scanning parameters

    Args:
        fullpath (str): fullpath to image file

        params (dict): parameters to provide to ImageScanner class

    Returns:
        numpy.array: raveled numpy array (1, n)
    '''
    img = Image.open(fullpath, 'r')
    patches = params.pop('patches')
    scan = ImageScanner(img, **params)
    patches = scan.random_scan(patches)
    X = np.array(patches.next())
    X = X.reshape(1, X.size)
    for patch in patches: 
        x = np.array(patch).ravel()
        x = x.reshape(1, x.size)
        X = np.append(X, x, axis=0)
    img.close()
    return X
# ------------------------------------------------------------------------------

def to_format(source, target, params, format_):
    '''
    generates patches for a single image and ouputs them to a file

    Args:
        source (str): fullpath to source file

        target (str): fullpath to target file

        params (dict): parameters to provide to ImageScanner class

        format_ (str):
            file format for output data
            options include: 'hdf', 'msgpack', 'json', 'csv'

    Returns:
        None
    '''
    data = get_data(source, params)
    data = DataFrame(data)
    
    func = getattr(data, 'to_' + format_)
    if format_ in ['hdf']:
        func(target, None)
    else:
        func(target)

def to_archive(source, target, format_, spec, params, processes=100):
    '''
    multiprocessed archiving tool for applyin to_format to files

    Args:
        source (str): fullpath to source file

        target (str): fullpath to target file

        format_ (str):
            file format for output data
            options include: 'hdf', 'msgpack', 'json', 'csv'

        spec (list): naming specification of files (dotslot syntax)

        params (dict): parameters to provide to ImageScanner class

        processes (Optional[int]):
            number of processes to spawn
            default: 100

    Returns:
        None
    '''
    source = get_info(source, spec).fullpath
    pool = multiprocessing.Pool(processes=processes)
    iterable = [(src, target, params, format_) for src in source]
    pool.map(to_format, iterable)
    pool.close()

def to_hdf_archive(source, target, spec, params):
    '''
    convert a directory of archived data into a single data archive file

    Args:
        source (str): fullpath to source directory of archive data

        target (str): fullpath to target directory

        spec (list): naming specification of files (dotslot syntax)

        params (dict): parameters to provide to ImageScanner class

    Returns:
        None
    '''

    info = get_info(source, spec)
    info['params'] = [params] * info.shape[0]
    format_ = info.extension.unique()
    if format_.size > 1:
        raise StandardError('multiple formats detected')
    format_ = format_[0]

    y = Series(info.index).apply(lambda x: [x]) * params['patches']
    y = list(chain(*y.tolist()))
    y = Series(y).astype(int)

    data = []
    for file_ in info.fullpath: 
        func = getattr(pd, 'read_' + format_)
        if format_ in ['hdf']:
            data.append(func(file_, None))
        else:
            data.append(func(file_))

    data = pd.concat(data, axis=0, ignore_index=True).astype(int)

    index = data.index.tolist()
    np.random.shuffle(index)
    y = y.reindex(index)
    y.reset_index(drop=True, inplace=True)
    data = data.reindex(index)
    data.reset_index(drop=True, inplace=True)
    X = data
    
    hdf = HDFStore(target)
    hdf['info'] = info
    hdf['X'] = X
    hdf['y'] = y
    hdf.close()
# ------------------------------------------------------------------------------

__all__ = ['execute_python_subshells', 'get_info', 'get_data', 'to_format',
           'to_archive', 'to_hdf_archive']

def main():
    pass

if __name__ == '__main__':
    help(main)