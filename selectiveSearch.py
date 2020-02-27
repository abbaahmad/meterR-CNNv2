#!/usr/bin/env python3
import cv2
import os
#from collections import namedtuple

def get_regions(imgpath,numToShow=100):
	'''
	returns list of rectangles (x,y,w,h) representing
	regions of interest(RoI) on image
	'''
	cv2.setUseOptimized(True)
	cv2.setNumThreads(4)

	#read image
	img = cv2.imread(imgpath)
	if img is None:
		print("Couldn't read image from {}".format(imgpath))
		return None


	#create Selective Search Segmentation Object using d
	selectS = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

	#set input image on which to run segmentation
	selectS.setBaseImage(img)

	#High recall but slow method
	selectS.switchToSelectiveSearchQuality()

	#run selective search segmentation
	rects = selectS.process()
	#print("[INFO] Total number of Region Proposals: {}".format(len(rects)))

	#Show regions
	count=0
	#Rect = namedtuple('Rect',['x','y','w','h'])
	Rects = rects[:int(numToShow)]
	#while count<numToShow:
		#Show original
	#imgOut = img.copy()
	
	#for i,rect in enumerate(rects):
	#	if(i< int(numToShow)):
	#		Rects.append(rect)
	#		x,y,w,h = rect
	#		cv2.rectangle(imgOut,(x,y),(x+w,y+h),(0,255,0),1,cv2.LINE_AA)
	#		cv2.imwrite("Ouput.png",imgOut)
	
	return Rects	
		
