#! /usr/bin/env python

from __future__ import print_function
import os
import shutil
from flask import Flask, render_template, request
from werkzeug import secure_filename

from core.pipeline import *
# ------------------------------------------------------------------------------

app = Flask('timbr', 
	static_folder=os.path.abspath('../web/static'),
	template_folder=os.path.abspath('../web/templates')
)

_UPLOAD_DIR = os.path.join(os.getcwd(), 'upload')
if os.path.exists(_UPLOAD_DIR):
	shutil.rmtree(_UPLOAD_DIR)
os.mkdir(_UPLOAD_DIR)

app.config['upload_dir'] = _UPLOAD_DIR

x = [{u'content': u' abura, bahia', u'heading': u'common name(s)'},
 {u'content': u' mitragyna\xa0spp. (sometimes listed as hallea or fleroya genera)',
  u'heading': u'scientific name'},
 {u'content': u' west and central africa', u'heading': u'distribution'},
 {u'content': u' 100-115 ft (30-35 m) tall, 3-5 ft (1-1.5 m) trunk diameter',
  u'heading': u'tree size'},
 {u'content': u' 35 lbs/ft3 (560 kg/m3)', u'heading': u'average dried weight'},
 {u'content': u' .45, .56', u'heading': u'specific gravity (basic, 12% mc)'},
 {u'content': u'\xa0820 lbf (3,670 n)', u'heading': u'janka hardness'},
 {u'content': u' 11,760 lbf/in2 (81.1 mpa)',
  u'heading': u'modulus of rupture'},
 {u'content': u' 1,386,000 lbf/in2 (9.56 gpa)',
  u'heading': u'elastic modulus'},
 {u'content': u' 6,220 lbf/in2 (42.9 mpa)', u'heading': u'crushing strength'},
 {u'content': u' radial: 4.3%, tangential: 9.2%, volumetric: 13.3%, t/r ratio: 2.1',
  u'heading': u'shrinkage'},
 {u'content': u' has a uniform yellow to pinkish-brown color, with sapwood indistinct from heartwood. ',
  u'heading': u'color/appearance'},
 {u'content': u' has a fine texture and a straight or slightly interlocked grain. ',
  u'heading': u'grain/texture'},
 {u'content': u' non-durable; poor resistance to decay or insect attack. good acid resistance. ',
  u'heading': u'rot resistance'},
 {u'content': u' takes glue and finishes well. has a slight blunting effect on cutting edges and tools due to a moderate silica content (.25%). ',
  u'heading': u'workability'},
 {u'content': u' has an unpleasant odor when freshly cut. ',
  u'heading': u'odor'},
 {u'content': u' has been known to cause allergic reactions including: nausea, eye irritation, giddiness, and vomiting. see the articles wood allergies and toxicity and wood dust safety for more information. ',
  u'heading': u'allergies/toxicity'},
 {u'content': u' seldom available in north america. price should be moderate when compared to other imported lumber. ',
  u'heading': u'pricing/availability'},
 {u'content': u'\xa0this wood species is not listed in the cites appendices, but\xa0is on the iucn red list. it is listed as\xa0vulnerable (under the hallea genus) due to a population reduction of over\xa020% in the past three generations, caused by a decline in its natural range, and exploitation. ',
  u'heading': u'sustainability'},
 {u'content': u' a general-purpose lumber used for furniture, interior millwork, plywood, and flooring. ',
  u'heading': u'common uses'},
 {u'content': u'\xa0sometimes sold under the name bahia, the handful of african species from the mitragyna genus that are sold interchangeably with one another include: m. ciliata, m. ledermannii, and m.\xa0stipulosa.\xa0these species have been formerly placed in the hallea genus (now considered a synonym), and\xa0fleroya. ',
  u'heading': u'comments'},
 {u'content': u' none available. ', u'heading': u'related species'}]

def predict(fullpath):
	data = { 
		'results': [
			{'source': '../static/img/wood.jpg', 'label': 'olive', 'confidence': '95%', 'description': x},
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