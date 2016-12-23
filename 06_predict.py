#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import random
import sys

import numpy as np
from PIL import Image
import six
import cPickle as pickle

import chainer
import numpy as np
import math
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import network

parser = argparse.ArgumentParser(
	description='Image inspection using chainer')
parser.add_argument('image', help='Path to inspection image file')
parser.add_argument('--model','-m',default='model', help='Path to model file')
parser.add_argument('--mean', default='mean.npy',
					help='Path to the mean file (computed by compute_mean.py)')
args = parser.parse_args()


#imageを読み込んで平均を引く、標準偏差で割る
def read_image(path, center=False, flip=False):
	#row, col, color -> color, row, col
	image = np.asarray(Image.open(path)).transpose(2, 0, 1) 
	#範囲を決める
	if center:
		top = left = cropwidth / 2
	else:
		top = random.randint(0, cropwidth - 1)
		left = random.randint(0, cropwidth - 1)
	bottom = model.insize + top
	right = model.insize + left

	image = image[:, top:bottom, left:right].astype(np.float32)
	#平均を引く
	image -= mean_image[:, top:bottom, left:right]
	#標準偏差で割る
	#devide by stdDeviation instead of by 256
	image/=g_stdDev
	#image/=255.0
	#左右反転
	#flip right and left 
	if flip and random.randint(0, 1) == 0:
		return image[:, :, ::-1] 
	else:
		return image



mean_image = pickle.load(open(args.mean, 'rb'))
g_stdDev = pickle.load(open("sigma.npy",'rb'))


model = network.imageModel()
serializers.load_hdf5("modelhdf5", model)
cropwidth = 256 - model.insize

model.to_cpu()
#model.to_gpu()



def predict(net,x):

	#x = chainer.Variable(x, volatile=False)
	#t = chainer.Variable(t, volatile=False)

	h = F.relu(model.conv1(x))
	h = F.max_pooling_2d(h, 3, stride=2)
	#print h.data.shape
	h = F.relu(model.conv2(h))
	h = F.max_pooling_2d(h, 3, stride=2)
	h = F.relu(model.conv3(h))
	h = F.max_pooling_2d(h, 3, stride=2)
	#h=F.relu(model.l1(h))
	h=F.dropout(F.relu(model.l1(h)),ratio=0.5,train=False)
	y=model.l2(h)
	
	return F.softmax(y)


#setattr(model, 'predict', predict)
img = read_image(args.image)
x = np.ndarray((1, 3, model.insize, model.insize), dtype=np.float32)
x[0]=img
#使用時にはvolatile='off'できる
x = chainer.Variable(np.asarray(x), volatile='off')

score = predict(model,x)
#score=cuda.to_cpu(score.data)
categories = np.loadtxt("labels.txt", str, delimiter="\t")

top_k = 20
prediction = zip(score.data[0].tolist(), categories)
prediction.sort(cmp=lambda x, y: cmp(x[0], y[0]), reverse=True)
for rank, (score, name) in enumerate(prediction[:top_k], start=1):
	print('#%d | %s | %4.1f%%' % (rank, name, score * 100))

