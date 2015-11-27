#! /usr/bin/env python

from itertools import *
from apiclient.discovery import build
from apiclient.errors import HttpError
# ------------------------------------------------------------------------------

class GoogleImageScraper(object):
	def __init__(self, key, cx, params):
		self._cx = cx
		self._cse = build('customsearch', 'v1', developerKey=key).cse()
		self._params = params
		self._response = []

	def get_response(self, num=10):
		params = copy(self._params)
		params['num'] = 1
		
		_pages = None
		try:
			_pages = service.cse().list(**params).execute()
		except HttpError:
			return output
		_pages = int(_pages['searchInformation']['totalResults'])
		
		pages = int(_pages / num)
		if _pages % num:
			pages += 1
		
		start = 1
		for page in xrange(1, pages):
			try:
				response = self._cse.list(start=start, **self._params).execute()
			except HttpError:
				break
			self._response.extend(response['items'])
			start += 1
# ------------------------------------------------------------------------------

__all__ = [
	'GoogleImageScraper'
]

def main():
    pass

if __name__ == '__main__':
    help(main)