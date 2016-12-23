#!/usr/bin/env python
import argparse
import os
import sys

import numpy
from PIL import Image
from PIL import ImageOps

import six.moves.cPickle as pickle


parser = argparse.ArgumentParser(description='Compute images mean array')
parser.add_argument('dataset', help='Path to training image-label list file')
parser.add_argument('--root', '-r', default='.',
					help='Root directory path of image files')
parser.add_argument('--output', '-o', default='mean.npy',
					help='path to output mean array')
args = parser.parse_args()

sum_image = None
count = 0
for line in open(args.dataset):# given tran.txt
	filepath = os.path.join(args.root, line.strip().split()[0])
	img=Image.open(filepath)
	img = ImageOps.grayscale(img)
	
	img.save(filepath)
	sys.stderr.write('\r{}'.format(count)) #print number of images?
	sys.stderr.flush()

sys.stderr.write('\n')



