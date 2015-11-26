import cPickle
from core.utils import *
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
    'description',
    'extension'
]

TEXT_SPEC = [
    'material',
    'text_id',
    'label',
    'origin',
    'extension'
]
# ------------------------------------------------------------------------------

class TextureClassifier(object):
	def __init__(self, model_path, db_path):
		self._model_path = model_path
		self._model = cPickle.load(model_path)
		self._image_path = os.path.join(db_path, 'image')
		self._text_path = os.path.join(db_path, 'text')
		self._temp_path = os.path.join(db_path, 'temp')

	@property
	def info(self):
		img = get_info(self._image_path, IMAGE_SPEC)
		txt = get_info(self._image_path, TEXT_SPEC)
		info = img.merge(txt, how='left', on='label')
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
	'TEXT_SPEC',
	'TextureClassifier'
]

def main():
    pass

if __name__ == '__main__':
    help(main)