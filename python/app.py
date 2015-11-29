#! /usr/bin/env python
'''
timbr wood classification flask app
'''
from __future__ import division, with_statement, print_function
from itertools import *
import time
import sys
import os
import shutil
import json
import cv2
from matplotlib import pyplot as plt
from flask import Flask, render_template, request
from werkzeug import secure_filename

from model import *
from core.utils import *
# ------------------------------------------------------------------------------

__MODEL = 'random-forest.019.aw-d_wood-mask.100x100.1-even.grid.pkl'
__ROOT = os.path.abspath('../web')
app = Flask('timbr', 
	static_folder=os.path.join(__ROOT, 'static'),
	template_folder=os.path.join(__ROOT, 'templates'),
)

upload_dir = os.path.join(__ROOT, 'uploads')
if os.path.exists(upload_dir):
	shutil.rmtree(upload_dir)
os.mkdir(upload_dir)

start = time.time()
print('loading model')
_MODEL = TextureClassifier(os.path.join(__ROOT, 'static/database'), __MODEL)
print('model loaded')
print('load time:', time.time() - start)

app.config['upload_dir'] = upload_dir
# ------------------------------------------------------------------------------

def save_histogram(source, destination, colorspace='rgb'):
	'''
	saves rgb or hsv hostograms of source image to destination

	Args:
		source (str):
			fullpath of source image
		
		destination (str):
			fullpath of destination image

		colorspace opt(str):
			colorspace of histogram
			acceptable values: 'rgb', 'hsv'
			default: 'rgb'
	
	Returns:
		None: None
	'''
	img = cv2.imread(source)
	if colorspace == 'hsv':
		img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hist = get_histograms(img, colorspace=colorspace)

	fig, ax = plt.subplots(figsize=(5.0, 3.5))

	for chan, data in hist.iteritems():
		label_lut = {
			'r': 'red',
			'g': 'green',
			'b': 'blue',
			'h': 'hue',
			's': 'saturation',
			'v': 'value'
		}

		color_lut = {
			'r': '#F77465',
			'g': '#A3C987',
			'b': '#5F95DE',
			'h': '#7EC4CF',
			's': '#AC92DE',
			'v': '#D1B58C'
		}
		ax.plot(data, label=label_lut[chan], color=color_lut[chan])
		# ax.legend()
		ax.set_frame_on(False)
		ax.set_xlim(0, 255)
		ax.get_yaxis().set_visible(False)
		ax.get_xaxis().set_visible(False)

	fig.tight_layout()
	fig.savefig(destination, transparent=True)

def predict(fullpath=None):
	'''
	predicts the material type of provided image

	Args:
		fullpath (str):
			fullpath of image file

	Returns:
		dict: dict of results
	'''
	results = []
	if fullpath == None:
		with open('../web/static/demo/demo.json', 'r') as file_:
			results = json.load(file_)
			fullpath = results[0]['source']
			save_histogram(fullpath, os.path.join(__ROOT, 'static/database/temp/rgb_hist.png'))
			save_histogram(fullpath, os.path.join(__ROOT, 'static/database/temp/hsv_hist.png'), colorspace='hsv')
	else:
		results = _MODEL.predict(fullpath)
	
	for item in results:
		src = os.path.split(item['source'])[-1]
		item['source'] = '../static/database/images/' + src
		item['confidence'] = round(item['confidence'] * 100, 1)
		desc = item['description']
		if isinstance(desc, list):
			item['description'] = item['description'][:min(len(desc)-1, 16)]
		else:
			item['description'] = []
	return {'results': results}


@app.route('/', methods=['GET', 'POST'])
def index():
	'''
	flask index.html endpoint
	'''
	if request.method == 'GET':
		data = predict()
		return render_template('index.html', **data)
	elif request.method == 'POST':
		files = [x for x in request.files.getlist('file[]')]
		f = files[0] #limit to first file
		fullpath = secure_filename(f.filename)
		fullpath = os.path.join(app.config['upload_dir'], fullpath)
		f.save(fullpath)
		data = predict(fullpath)

		# generate histograms
		temp = os.path.join(__ROOT, 'static/database/temp')
		if os.path.exists(temp):
			shutil.rmtree(temp)
		os.mkdir(temp)

		save_histogram(fullpath, os.path.join(__ROOT, 'static/database/temp/rgb_hist.png'))
		save_histogram(fullpath, os.path.join(__ROOT, 'static/database/temp/hsv_hist.png'), colorspace='hsv')

		return render_template('index.html', **data)
		
if __name__ == '__main__':
	app.run(debug=True)