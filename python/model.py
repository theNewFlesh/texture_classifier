import os
import cPickle
import json
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
# ------------------------------------------------------------------------------

class TextureClassifier(object):
	def __init__(self, db_path):
		self._db_path = db_path
		self._model_path = os.path.join(db_path, 'models')
		self._image_path = os.path.join(db_path, 'images')
		self._desc_path = os.path.join(db_path, 'descriptions.json')
		self._temp_path = os.path.join(db_path, 'temp')
		self.set_model(filename)		
	
	def set_model(self, filename):
		fullpath = os.path.join(self._model_path, filename)
		with open(fullpath, 'r') as model:
			self._model = cPickle.load(model)

	@property
	def info(self):
		desc = None
		with open(self._desc_path, 'r') as d:
			desc = json.load(d)

		info = get_info(self._image_path, IMAGE_SPEC)
		info['description']	= info.label.apply(
			lambda x: desc[x] if desc.has_key(x.lower()) else None)	
		return info

	def get_data(self, fullpath):
		spec = ['name', 'extension']
		info = get_info(fullpath, spec)
		info['label'] = 'unknown'
		info['params'] = None
		info.params = info.params.apply(lambda x: PARAMS)
		data = process_data(info).drop('y', axis=1)
		return data

	def get_results(self, pred):
		data = pred.merge(self.info, how='inner', on='label')
		data.drop_duplicates('label', inplace=True)
		data = data.apply(lambda x: x.to_dict(), axis=1).tolist()
		return data

	def predict(self, filepath):
		pred = self.get_data(filepath)
		pred = self._model.predict(pred)
		pred = compile_predictions(pred)
		pred = self.get_results(pred)
		return pred
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