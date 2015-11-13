import os
import re
from itertools import *
from copy import copy
import multiprocessing
import numpy as np
from pandas import DataFrame
from PIL import Image
from core.image_scanner import ImageScanner
# ------------------------------------------------------------------------------

def get_series_info(root, spec, ignore=['\.DS_Store']):
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
# ------------------------------------------------------------------------------

# defined in module body for multiprocessing
def _get_data(fullpath, patches, params):
    img = Image.open(fullpath, 'r')
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

def _to_msgpack(args):
    fullpath, patches, params = args
    data = _get_data(fullpath, patches, params)
    base, filename = os.path.split(fullpath)
    filename = os.path.splitext(filename)[0] + '.msgpack'
    base = os.path.split(base)[0]
    output = os.path.join(base, 'msgpack', filename)
    DataFrame(data).to_msgpack(filename)

def archive_data(info, params, cores=100):
    patches = params.pop('patches')
    pool = multiprocessing.Pool(processes=cores)
    pool.map(_to_msgpack, map(lambda x: (x, patches, params), info.fullpath))
    pool.close()
# ------------------------------------------------------------------------------

__all__ = ['get_series_info', 'archive_data']

def main():
    pass

if __name__ == '__main__':
    help(main)