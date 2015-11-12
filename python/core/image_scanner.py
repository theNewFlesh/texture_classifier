from __future__ import division
from itertools import *
# import multiprocessing
import numpy as np
# ------------------------------------------------------------------------------

class ImageScanner(object):
    def __init__(self, image, aspect_ratio=None, min_size=0.1, max_size=0.2,
                 patch_resolution=None, resample=0, rotation=None): #, cores=24):
        # generate maximum patch aspect
        x, y = None, None
        if not aspect_ratio:
            x, y = image.size
        else:
            x, y = image.size
            if aspect_ratio >= 1:
                x = int(y * aspect_ratio)
            else:
                y = int(x / aspect_ratio)

        self._image = image
        self._min_resolution = int(min_size * x), int(min_size * y)
        self._max_resolution = int(max_size * x), int(max_size * y)
        self._resample = resample
        self._rotation = rotation
        self._patch_resolution = None
        # self._cores = cores
        if patch_resolution != None:
            if not isinstance(patch_resolution, str):
                self._patch_resolution = patch_resolution
            elif patch_resolution is 'auto':
                self._patch_resolution = self._min_resolution
    # --------------------------------------------------------------------------

    @property
    def _variables(self):
        '''
        conveinience method for returning oft used attributes
        '''
        return (
            self._image,
            self._min_resolution,
            self._max_resolution
        )

    def _even_resolutions(self, patches):
        '''
        generate a list of evenly spaced patch resolutions
        '''
        img, min_, max_ = self._variables

        x = np.linspace(min_[0], max_[0], patches, dtype=int)
        y = np.linspace(min_[1], max_[1], patches, dtype=int)
        return izip(x, y)

    def _random_resolutions(self, patches):
        '''
        generate a list of randomly spaced patch resolutions
        '''
        img, min_, max_ = self._variables
        
        # steps should be set to the patch size range of lesser of
        # max_resolution's two dimensions
        steps = max_[0] - min_[0]
        if max_[0] > max_[1]:
            steps = max_[1] - min_[1]
        x = np.linspace(min_[0], max_[0], steps, dtype=int)
        y = np.linspace(min_[1], max_[1], steps, dtype=int)

        for index in np.random.choice(steps, size=patches, replace=True):
            yield x[index-1], y[index-1]

    def _get_patch(self, bbox):
        img = self._image.crop(bbox)
        rotations = [0, 90, 180, 270]
        if self._patch_resolution:
            img = img.resize(self._patch_resolution, self._resample)
        if self._rotation:
            if self._rotation == 'random':
                img = img.rotate( np.random.choice(rotations) )
            elif self._rotation in rotations:
                img = img.rotate(self._rotation)
        return img
    # --------------------------------------------------------------------------

    def get_resolutions(self, patches=10, resolutions='even'):
        '''
        generates a list of patch resolutions
        '''
        if resolutions == 'even':
            return self._even_resolutions(patches)
        elif resolutions =='random':
            return self._random_resolutions(patches)
        
    def _grid_scan(self, resolution):
        '''
        scan entire area of image given a sample resolution
        '''
        img, min_, max_ = self._variables
        
        bbox_x, bbox_y = img.getbbox()[-2:]
        x_sample, y_sample = resolution
        x_scans = int(bbox_x / x_sample)
        y_scans = int(bbox_y / y_sample)
        for row in xrange(y_scans):
            upper = y_sample * row
            lower = upper + y_sample
            for col in xrange(x_scans):
                left = x_sample * col
                right = left + x_sample
                bbox = (left, upper, right, lower)
                yield self._get_patch(bbox)

    def grid_scan(self, patches=10, resolutions='even'):
        '''
        scans entire image in a grid-like fashion
        '''
        rez = self.get_resolutions(patches, resolutions)
        output = map(self._grid_scan, rez)
        return chain.from_iterable(output)

    def random_scan(self, patches=100):
        '''
        generates random patches of image
        '''
        img, min_, max_ = self._variables

        for x1, y1 in self._random_resolutions(patches):
            x, y = img.size
            left = np.random.choice(range(x - x1))
            right = left + x1
            upper = np.random.choice(range(y - y1))
            lower = upper + y1
            bbox = (left, upper, right, lower)
            yield self._get_patch(bbox)
# ------------------------------------------------------------------------------

__all__ = ['ImageScanner']

def main():
    pass

if __name__ == '__main__':
    help(main)