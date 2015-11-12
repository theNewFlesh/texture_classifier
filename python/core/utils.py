import os
import re
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
def __get_data(args):
    fullpath, patches, params = args
    img = Image.open(fullpath)
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

def get_data(info, params, cores=100):
    patches = params.pop('patches')
    pool = multiprocessing.Pool(processes=cores)
    data = pool.map(__get_data, imap(lambda x: (x, patches, params), info.fullpath))
    data = np.concatenate(data, axis=0)
    return data
# ------------------------------------------------------------------------------

__all__ = ['get_series_info', 'get_data']

def main():
    pass

if __name__ == '__main__':
    help(main)