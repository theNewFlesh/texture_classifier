#! /usr/bin/env python
'''
contains the WoodDatabaseScraper class used for scraping wood-database.com
'''
from __future__ import division, with_statement, print_function
from itertools import *
import os
import re
import urllib
from bs4 import BeautifulSoup
import requests
import pandas as pd
from pandas import DataFrame, Series
# ------------------------------------------------------------------------------

class WoodDatabaseScraper(object):
	def __init__(self):
		'''
		scrapes wood-database.com for images and image descriptions
		'''
		self._url = 'http://www.wood-database.com'

	def get_wood_urls(self):
		'''
		yields links of each different wood page

		Yields:
			str: url
		'''
		response = requests.get(self._url).content
		soup = BeautifulSoup(response)
		x = soup.select('.fusion-column-wrapper')
		x = list(chain(*[x.select('a') for x in x]))
		x = x[27:604]
		for item in x:
			try:
				yield item.attrs['href']
			except:
				pass

	def get_image_links(self, urls):
		'''
		lists all wood texture urls for wood-database.com

		Args:
			urls (iterable):
				links generated by get_wood_urls

		Returns:
			list: urls
		'''
		links = []
		for i, url in enumerate(urls):
			response = requests.get(url).content
			soup = BeautifulSoup(response)
			links = []
			for item in soup.select('.wp-caption.aligncenter a'):
				links.append(item.attrs['href'])
			
			print('{:<5}{}'.format(str(i), url))

		return list(chain(*links))
	
	def write_images(self, root, links, prefix=None):
		'''
		writes all images to a given root directory

		Args:
			root (str):
				root directory

			links (list):
				list of image links provided by get_image_links

			prefix opt(str):
				filename prefix
				default: None

		Returns:
			None: None
		'''
		for link in links:
			filename = link.split('/')[-1]
			if prefix:
				filename = prefix + filename
			fullpath = os.path.join(root, filename)
			urllib.urlretrieve(link, fullpath)

	def scrape_images(self, root, prefix=None):
		'''
		scrapes and saves all texture images from wood-database.com

		Args:
			root (str): directory to save images in

			prefix opt(str):
				filename prefix
				default: None
		'''
		urls = self.get_wood_urls()
		links = self.get_image_links(urls)
		self.write_images(root, links, prefix=prefix)

	def _clean_description(self, element):
		'''
		cleans up aggregated descriptions
		'''
		data = [x.getText() for x in element]
		data = Series(data)
		data = data.apply(lambda x: re.sub('\n', ' ', x))
		mask = data.apply(lambda x:
			False if re.search('\(sanded|\(sealed|\(endgrain|\(curl|\(burl|^$', x) else True)
		data = data[mask]
		
		def func(item):
			try:
				return list(re.search('(.*?):(.*)', item).groups())
			except:
				return [item, None]
		
		data = data.apply(func).tolist()
		data = DataFrame(data, columns=['heading', 'content'])
		
		mask = data.content.apply(lambda x: pd.notnull(x))
		if mask.shape[0] > 0:
			mask.ix[0] = True
		data = data[mask]
		return data

	def get_descriptions(self, urls):
		'''
		gets description data for each wood type

		Args:
			urls (iterable):
				links generated by get_wood_urls

		Returns:
			dict: dict of lists
		'''
		data = {}
		for url in urls:
			response = requests.get(url).content
			soup = BeautifulSoup(response)

			desc1 = soup.select('.post-content table tbody tr td p')
			desc2 = soup.select('.post-content blockquote')

			desc1 = self._clean_description((desc1))
			desc2 = self._clean_description((desc2))
			datum = pd.concat([desc1, desc2], axis=0)
			datum = datum.apply(lambda x: x.to_dict(), axis=1).tolist()

			name = re.split('/', url)[-2]
			data[name] = datum
		return data
# ------------------------------------------------------------------------------

__all__ = [
	'WoodDatabaseScraper',
]

def main():
	pass

if __name__ == '__main__':
	help(main)