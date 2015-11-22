from __future__ import division
from itertools import *
import numpy as np
# ------------------------------------------------------------------------------

class ImageScanner(object):
    '''Used for scanning images and producting image pathches through various techniques'''
    def __init__(self, image, min_resolution=(100, 100), max_resolution=(200, 200),
                 patch_resolution=None, resample=0, rotation=None, **kwargs):
        '''
        Args:
            image (PIL.Image): Python Imaging Library Image object
            
            min_resolution (Optional[tuple]):
                minimum sampling size
                default: (100, 100)

            max_resolution (Optional[tuple]):
                maximum sampling size
                default: (200, 200)

            patch_resolution (Optional[tuple]):
                output patch resolution (x, y)
                default: None

            resample (Optional[str]):
                resampling tfilter used by PIL.Image
                options include:
                    `PIL.Image.NEAREST`  (use nearest neighbour)
                    `PIL.Image.BILINEAR` (linear interpolation)
                    `PIL.Image.BICUBIC`  (cubic spline interpolation)
                    `PIL.Image.LANCZOS`  (a high-quality downsampling filter)
                default: 0

            rotation (Optional[int or str]):
                degree of rotation to be applied to output patches
                options include: 0, 90, 180, 270, 'random'
                default: None
        '''
        self._image = image
        self._min_resolution = min_resolution
        self._max_resolution = max_resolution
        self._patch_resolution = min_resolution
        if patch_resolution:
            self._patch_resolution = patch_resolution
        self._resample = resample
        self._rotation = rotation

        # convenience attribute
        self.__vars = self._image, self._min_resolution, self._max_resolution
    # --------------------------------------------------------------------------

    def _even_resolutions(self, patches):
        '''
        generate a list of evenly spaced patch resolutions
        '''
        img, min_, max_ = self.__vars

        x = np.linspace(min_[0], max_[0], patches, dtype=int)
        y = np.linspace(min_[1], max_[1], patches, dtype=int)
        return izip(x, y)

    def _random_resolutions(self, patches):
        '''
        generate a list of randomly spaced patch resolutions
        '''
        img, min_, max_ = self.__vars
        
        # steps should be set to the patch size range of lesser of
        # max_resolution's two dimensions
        steps = max_[0] - min_[0]
        if max_[0] > max_[1]:
            steps = max_[1] - min_[1]
        if steps == 0:
            steps = 1
        x = np.linspace(min_[0], max_[0], steps, dtype=int)
        y = np.linspace(min_[1], max_[1], steps, dtype=int)

        for index in np.random.choice(steps, size=patches, replace=True):
            yield x[index-1], y[index-1]

    def _get_patch(self, bbox):
        '''
        return a cropped (resized amd/or rotated) image based upon bounding box
        '''
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

    def get_resolutions(self, num=10, spacing='even'):
        '''
        generates a list of patch resolutions

        Args:
            num (Optional[int]):
                number of resolutions returned
                default: 10

            spacing (Optional[str]):
                spacing between resolution sizes
                options include: 'even', 'random'
                default: 'even'

        Yields:
            tuple: (x, y) resolution
        '''
        if spacing == 'even':
            return self._even_resolutions(num)
        elif spacing =='random':
            return self._random_resolutions(num)

    def grid_scan(self, resolutions=10, spacing='even', **kwargs):
        '''
        scans entire image in a grid-like fashion

        Args:
            resolutions (Optional[int]):
                number of sampling patch resolutions to return
                a single grid produces multiple patches (image / sampling resolution)
                default: 10

            spacing (Optional[str]):
                spacing between resolution sizes
                options include: 'even', 'random'
                default: 'even'

        Yields:
            PIL.Image: cropped (resized and/or rotated) patch
        '''
        def _grid_scan(resolution):
            '''
            scan entire area of image given a sample resolution
            '''
            img, min_, max_ = self.__vars
            
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

        rez = self.get_resolutions(patches, spacing)
        output = map(_grid_scan, rez)
        return chain.from_iterable(output)

    def random_scan(self, patches=100, **kwargs):
        '''
        generates patches of random sample size and location from image

        Args:
            patches (Optional[int]):
                number of patches returned
                default: 100

        Yields:
            PIL.Image: cropped (resized and/or rotated) patch
        '''
        img, min_, max_ = self.__vars

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