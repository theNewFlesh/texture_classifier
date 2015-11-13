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

def execute_python_subshells(iterable, script, args=[]):
    cmd = [
        'for f in', ' '.join(iterable), ';',
            'do (python', script, ' '.join(args), '$f 2>/dev/null &);',
        'done'
    ]
    cmd = ' '.join(cmd)
    subprocess.call(cmd, shell=True, stdout=PIPE, stderr=PIPE)
# ------------------------------------------------------------------------------

def get_info(root, spec, ignore=['\.DS_Store']):
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
    '''
    data = get_data(source, params)
    data = DataFrame(data)

    base, filename = os.path.split(source)
    filename = os.path.splitext(filename)[0] + '.' + format_
    output = os.path.join(target, filename)
    
    func = getattr(data, 'to_' + format_)
    if format_ in ['hdf']:
        func(output, None)
    else:
        func(output)

def to_archive(source, target, format_, spec, params, cores=100):
    source = get_info(source, spec).fullpath
    pool = multiprocessing.Pool(processes=cores)
    iterable = [(src, target, params, format_) for src in source]
    pool.map(to_format, iterable)
    pool.close()

def to_hdf_archive(source, target, spec, params):
    info = get_info(source, spec)
    info['params'] = [params] * info.shape[0]
    format_ = info.extension.unique()
    if len(format_) > 1:
        raise StandardError('multiple formats detected')
    format_ = format_[0]

    y = Series(info.index).apply(lambda x: [x]) * params['patches']
    y = list(chain(*y.tolist()))
    y = Series(y)

    data = []
    for file_ in info.fullpath: 
        func = getattr(pd, 'read_' + format_)
        if format_ in ['hdf']:
            data.append(func(file_, None))
        else:
            data.append(func(file_))

    data = pd.concat(data, axis=0)
    data['y'] = y

    index = data.index
    np.random.shuffle(index.tolist())
    data = data.ix[index]
    X = data.drop('y', axis=1)
    y = data['y']
    
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