#!/usr/bin/env python

import cv2
import sys
import caffe

model = "./deploy.prototxt"
weights = "./weights_single_frame.caffemodel"

caffe.set_mode_cpu()

net = caffe.Net(model, weights, caffe.TEST)

f = "../test.txt"
nnInputWidth = 32
nnInputHeight = 32
f = open(f, 'r')
total = 0
num = 0
while True:
	line = f.readline()
	if not line:
		break
	l = line.split()
#	print l
#	print(l[0])
	image = cv2.imread(l[0])
#	cv2.imshow('image',image)
#	cv2.waitKey(0)
	inputResImg = cv2.resize(image, (nnInputWidth, nnInputHeight), interpolation=cv2.INTER_CUBIC)
	transposedInputImg =inputResImg.transpose(2,0,1)
	net.blobs['data'].data[...]=transposedInputImg
	out = net.forward()
	max_value=0
	max_iter=-1
	total = total + 1
	for y in out:
		scores = out[y]
		for x in range(len(scores[0])):
			k = scores[0][x]
			if k > max_value:
				max_value = k
				max_iter = x
				if max_iter== int(l[1]):
#					print "YES"
					num = num +1
#		print scores, max_iter
#		print scores
#	print "\n"

print "Accuracy: ", num*1.0/total
