#! /usr/bin/env python
'''
contains the TextureClassifier class used for predicting the material type of a supplied texture
'''
from __future__ import division, with_statement, print_function
from itertools import *
import os
import time
import cPickle
import json
import pandas as pd
from core.pipeline import *
# ------------------------------------------------------------------------------

PARAMS = {
	'scan_method':      'grid_scan',
	'min_resolution':   (100, 100),
	'max_resolution':   (100, 100),
	'resolutions':      1,
	'spacing':          'even',
	# 'patch_resolution': (100, 100),
	# 'normalize':        True,
	'bins':             256
}

IMAGE_SPEC = [
	'material',
	'image_id',
	'label',
	'origin',
	'descriptor',
	'extension'
]

SEP = '\.'
# ------------------------------------------------------------------------------

class TextureClassifier(object):
	def __init__(self, db_path, model_name):
		'''
		high level texture classifier

		Args:
			db_path (str):
				fullpath to database

			model_name (str):
				name of .pkl model found in db_path/models
		'''
		self._db_path = db_path
		self._model_path = os.path.join(db_path, 'models')
		self._image_path = os.path.join(db_path, 'images')
		self._desc_path = os.path.join(db_path, 'descriptions.json')
		self._temp_path = os.path.join(db_path, 'temp')
		self.set_model(model_name)      
	
	def set_model(self, filename):
		'''
		sets TextureClassifier's internal model to given .pkl model

		Returns:
			None: None
		'''
		fullpath = os.path.join(self._model_path, filename)
		with open(fullpath, 'r') as model:
			self._model = cPickle.load(model)
		
	@property
	def info(self):
		'''
		compiled information about data in db_path/images and db_path/descriptions.json

		Returns:
			DataFrame: info object
		'''
		desc = None
		with open(self._desc_path, 'r') as d:
			desc = json.load(d)

		info = get_info(self._image_path, IMAGE_SPEC, sep=SEP)
		info['description'] = info.label.apply(
			lambda x: desc[x] if desc.has_key(x.lower()) else None) 
		return info

	def get_data(self, fullpath):
		'''
		processes image file

		Args:
			fullpath (str):
				fullpath to image file

		Returns:
			DataFrame: data
		'''
		info = get_info(fullpath)
		info['label'] = 'unknown'
		info['params'] = None
		info.params = info.params.apply(lambda x: PARAMS)
		data = process_data(info).drop('y', axis=1)
		return data

	def get_results(self, pred):
		'''
		converts predictions into results

		Args:
			pred (numpy.array):
				output of self._model.predict(data)

		Returns:
			list (of dicts): results
		'''
		data = pred.merge(self.info, how='inner', on='label')
		data.drop_duplicates('label', inplace=True)
		data = data.apply(lambda x: x.to_dict(), axis=1).tolist()
		return data

	def predict(self, filepath):
		'''
		predict the material type of provided image file

		Args:
			fullpath (str):
				fullpath to image file

		Returns:
			list (of dicts): results
		'''
		pred = self.get_data(filepath)
		pred = self._model.predict(pred)
		pred = compile_predictions(pred)
		pred = self.get_results(pred)
		return pred

	def classification_report(self, info):
		'''
		process each image in info and compile a classification report for all of them

		Args:
			info (DataFrame):
				info object which lists images to be processed

		Returns:
			DataFrame: classification report
		'''
		data = []
		for i, row in info.iterrows():
			filepath = row['source']
			pred = self.get_data(filepath)
			pred = self._model.predict(pred)
			pred = compile_predictions(pred)
			pred = pred.head(1)
			pred['origin'] = row['origin']
			pred['ytrue'] = row['label']
			data.append(pred)
		data = pd.concat(data, axis=0, ignore_index=True)
		data.columns = ['confidence', 'ypred', 'origin', 'ytrue']
		data = data[['origin', 'confidence', 'ytrue', 'ypred']]
		return data
# ------------------------------------------------------------------------------

__all__ = [
	'PARAMS',
	'IMAGE_SPEC',
	'TextureClassifier'
]

def main():
	pass

if __name__ == '__main__':
	help(main)