#! /usr/bin/env python

import os
import sys
from core.utils import _get_data
from pandas import DataFrame
from datetime import datetime

def main(fullpath):
	print datetime.now(), 'STARTED:', fullpath
	params = {
	            'aspect_ratio':     1,
	            'min_size':         0.05,
	            'max_size':         0.1,
	            'patches':          1000,
	            'patch_resolution': (100, 100),
	            'rotation':         'random'
	}
	patches = params.pop('patches')

	base, filename = os.path.split(fullpath)
	filename = os.path.splitext(filename)[0]
	outpath = os.path.split(base)[0]
	outpath = os.path.join(outpath, 'msgpack')

	for msgpack in os.listdir(outpath):
		if filename in msgpack:
			return

	filename += '.msgpack'
	outpath = os.path.join(outpath, filename)

	data = _get_data(fullpath, patches, params)
	DataFrame(data).to_msgpack(outpath)
	print datetime.now(), 'COMPLETE:', outpath

if __name__ == '__main__':
	main(sys.argv[1])