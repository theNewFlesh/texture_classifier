#! /usr/bin/env python

from __future__ import print_function
import os
import shutil
from flask import Flask, render_template, request
from werkzeug import secure_filename

# from core.pipeline import *
# ------------------------------------------------------------------------------

app = Flask(__name__)

_UPLOAD_DIR = os.path.join(os.getcwd(), 'upload')
if os.path.exists(_UPLOAD_DIR):
	shutil.rmtree(_UPLOAD_DIR)
os.mkdir(_UPLOAD_DIR)

app.config['upload_dir'] = _UPLOAD_DIR

def predict(fullpath):
	data = { 
		'results': [
			{'source': '../static/img/wood.jpg', 'label': 'olive', 'confidence': '95%', 'description': 'olive wood'},
			{'source': '../static/img/wood.jpg', 'label': 'black adler', 'confidence': '39%'},
			{'source': '../static/img/wood.jpg', 'label': 'african mahogany', 'confidence': '34%'},
			{'source': '../static/img/wood.jpg', 'label': 'sapele', 'confidence': '7%'},
			{'source': '../static/img/wood.jpg', 'label': 'kingwood', 'confidence': '2%'},
			{'source': '../static/img/wood.jpg', 'label': 'birch', 'confidence': '1%'}
		]
	}
	return data

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'GET':
		data = predict(None)
		return render_template('index.html', **data)
	elif request.method == 'POST':
		files = [x for x in request.files.getlist('file[]')]
		f = files[0] #limit to first file
		fullpath = secure_filename(f.filename)
		fullpath = os.path.join(app.config['upload_dir'], fullpath)
		f.save(fullpath)
		data = predict(fullpath)
		return render_template('index.html', **data)
		
if __name__ == '__main__':
	app.run(debug=True)