#! /usr/bin/env python

import sys
import os
import subprocess
from subprocess import PIPE
from core.utils import get_series_info

def main(script, root):
	spec = ['texture', 'image_id', 'image_class', 'common_name', 'pass_', 'extension']
	info = get_series_info(root, spec)
	info = info[(info.image_class == 'a') & (info.pass_ == 'diffuse')]
	args = ' '.join(info.fullpath.tolist())
	cmd = 'for f in ' + args + '; do (python ' + script + ' $f 2>/dev/null &); done'
	subprocess.call(cmd, shell=True, stdout=PIPE, stderr=PIPE)

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])