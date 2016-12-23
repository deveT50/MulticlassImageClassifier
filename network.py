#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import math
import chainer.initializers
#http://qiita.com/icoxfog417/items/5fd55fad152231d706c2
#Convolutional Layer: 特徴量の畳み込みを行う層
#Pooling Layer: レイヤの縮小を行い、扱いやすくするための層
#まずPooling Layerですが、これは画像の圧縮を行う層になります。画像サイズを圧縮して、後の層で扱いやすくできるメリットがあります。このPoolingを行う手段として、Max Poolingがあります。これは、各領域内の最大値をとって圧縮を行う方法です。

#scikit-learnのpreprocessing.scaleが便利です。

#Fully Connected Layer: 特徴量から、最終的な判定を行う層
#DCGAN
#http://qiita.com/rezoolab/items/5cc96b6d31153e0c86bc
#batchNormalization
#http://hirotaka-hachiya.hatenablog.com/entry/2016/08/06/175824
#http://www.iandprogram.net/entry/2016/02/11/181322
#http://qiita.com/bohemian916/items/9630661cd5292240f8c7
class imageModel(chainer.Chain):
	#MSRA scaling について	
	#http://qiita.com/dsanno/items/47f52d6f6070ad9847e1

	#insize = 128
	insize = 227
	global w
	w = math.sqrt(2)  # MSRA scaling


	def __init__(self):
		initializer = chainer.initializers.HeNormal()
		super(imageModel, self).__init__(
			#入力チャネル,出力チャネル, フィルタサイズpx
			#209*209が出力チャネル枚
			#Network in Network <http://arxiv.org/abs/1312.4400v3>
			#60.9%モデル--------------------------------------
			#conv1=L.Convolution2D(3, 8, 7),
			#conv2=L.Convolution2D(8, 16, 5),
			#conv3=L.Convolution2D(16, 32, 3),
			#conv4=L.Convolution2D(32, 48, 3),


			conv1=L.Convolution2D(3, 8, 7,wscale=w),
			conv2=L.Convolution2D(8, 16, 5,wscale=w),
			conv3=L.Convolution2D(16, 32, 3,wscale=w),
			conv4=L.Convolution2D(32, 48, 3,wscale=w),


			
			#-----------------------------------------vasilyモデル
			#conv1 = F.Convolution2D(  3,  64, 4, stride = 2, pad = 1, initialW=initializer),
			#conv2 = F.Convolution2D( 64, 128, 4, stride = 2, pad = 1, initialW=initializer),
			#conv3 = F.Convolution2D(128, 256, 4, stride = 2, pad = 1, initialW=initializer),
			#conv4 = F.Convolution2D(256, 512, 4, stride = 2, pad = 1, initialW=initializer),
			#fl	= L.Linear(100352, 2, initialW=initializer),
			#bn1   = F.BatchNormalization(64),
			#bn2   = F.BatchNormalization(128),
			#bn3   = F.BatchNormalization(256),
			#bn4   = F.BatchNormalization(512))


		)
		self.train = True
	def __call__(self, x, t, train):
		
		"""
		h = F.max_pooling_2d(F.relu(self.mlpconv1(x)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(self.mlpconv2(h)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(self.mlpconv3(h)), 3, stride=2)
		h = self.mlpconv4(F.dropout(h, train=self.train))
		y = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))
		"""
		"""
		ninのまね
		h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
		h = F.max_pooling_2d(F.relu(self.conv3(h)), 3, stride=2)
		h = self.conv4(F.dropout(h, train=self.train))
		y = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0], 1000))
		"""

		#---------------------------------------------60model
		#h = self.conv1(x)
		#h = F.relu(h)
		#h = F.max_pooling_2d(h, 3, stride=2)

		#h = self.conv2(h)
		#h = F.relu(h)
		#h = F.average_pooling_2d(h, 3, stride=2)

		#h = self.conv3(h)
		#h = F.relu(h)
		#h = F.average_pooling_2d(h, 3, stride=2)
		
		#h = self.conv4(h)
		#h = F.relu(h)
		#h = F.average_pooling_2d(h, 3, stride=2)
		
		#y = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0],48))

		#----------------------------------------vasily
		#h = F.relu(self.bn1(self.conv1(x)))
		#h = F.relu(self.bn2(self.conv2(h)))
		#h = F.relu(self.bn3(self.conv3(h)))
		#h = F.relu(self.bn4(self.conv4(h)))
		#y = self.fl(h)

		h = self.conv1(x)
		h = F.relu(h)
		h = F.max_pooling_2d(h, 3, stride=2)

		h = self.conv2(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)

		h = self.conv3(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)
		
		h = self.conv4(h)
		h = F.relu(F.dropout(h, ratio=0.5,train=train))
		h = F.average_pooling_2d(h, 3, stride=2)
		
		y = F.reshape(F.average_pooling_2d(h, 6), (x.data.shape[0],48))


		#http://qiita.com/supersaiakujin/items/ccdb41c1f33ad5d27fdf
		#活性化関数softmax=exp(a)/sum(exp(a))は、何でもありの入力の値を確率に直す
		if train:
			#cross_entropyは誤差関数。すべてのラベルについて-sum(log(softmax(y))*Label)する。これを最小化したい。
			return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
		else:
			return F.accuracy(y, t)




