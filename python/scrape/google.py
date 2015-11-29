#! /usr/bin/env python
'''
contains the GoogleImageScraper class used for scraping google images
'''
from __future__ import division, with_statement, print_function
from itertools import *
from apiclient.discovery import build
from apiclient.errors import HttpError
# ------------------------------------------------------------------------------

class GoogleImageScraper(object):
	'''
	scrapes google images
	'''
	def __init__(self, key, cx, params):
		'''
		Args:
			key (str):
				google api key

			cx (str):
				google cse key

			params (dict):
				google image search parameters
		'''
		self._cx = cx
		self._cse = build('customsearch', 'v1', developerKey=key).cse()
		self._params = params
		self._response = []

	@property
	def response(self):
		'''
		google image search response

		Returns:
			dict: response
		'''
		return self._response

	def issue_query(self):
		'''
		issue google image query
		
		Returns:
			None: None
		'''
		num = 10
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