import os
import cv2

def iou_region(boxA,boxB):
	'''
	returns area of intersection over union of 2 boxes A & B
	'''
	#coordinates of intersecting rect
	x1max = max(boxA[0],boxB[0])
	y1max = max(boxA[1],boxB[1])
	x2max = max(boxA[2],boxB[2])
	y2max = max(boxA[3],boxB[3])
	
	#area of intersecting rect
	#"+1" is cause pixel indices start from 0 and include the last available pixel
	#to get correct area, "+1" is added to all sides
	interArea = max(0,x2max-x1max + 1) * max(0,y2max-y1max + 1)
	
	#areas of both boxes
	boxAArea = (boxA[2]-boxA[0] + 1) *(boxA[3]-boxA[1] + 1)
	boxBArea = (boxB[2]-boxB[0] + 1) *(boxB[3]-boxB[1] + 1)
	
	unionArea = float(boxAArea + boxBArea - interArea)
	
	if unionArea == 0:
		return 0.0
	
	iou = interArea/unionArea
	#print("[INFO] IoU is {}".format(iou))
	
	return iou

def crop_image(image,x1,y1,x2,y2):
	'''
	returns region for alexnet classification
	'''
	assert (x2>x1 and y2>y1), "Dimensions are wrong"
	
	return image[y1:y2,x1:x2]

def resize_image(image,width,height):
	'''
	image.shape = [height,width]
	returns resized image of given dimensions
	'''
	
	aspect_ratio = width/ image.shape[1]
	dim = (width,int(image.shape[0]*aspect_ratio))
	
	resized = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
	
	return resized

def calculate_dims(label_list,img_width=100,img_height=100):
	'''
	converts YOLO format to bounding box coordinates 
	of form x1,y1,x2,y2
	'''
	if len(label_list) != 4:
		print("[WARNING] Length of label is {}".format(len(label_list)))
		raise AssertionError("List doesn't contain all values")
	#elif img_width < 450 and img_height <450:
	#	print("Width "+str(img_width)+ " Height "+ str(img_height))
	#	raise AssertionError("Image already cropped")
	
	x = float(label_list[0])
	y = float(label_list[1])
	width = float(label_list[2])
	height = float(label_list[3])
	
	coordinates = []
	x1 = (x-(width/2))*img_width
	y1 = (y-(height/2))*img_height
	x2 = (x+(width/2))*img_width
	y2 = (y+(height/2))*img_height
	#print("x1:{}, y1:{}, x2:{}, y2:{}".format(x1,y1,x2,y2))
	
	coordinates.append(x1)
	coordinates.append(y1)
	coordinates.append(x2)
	coordinates.append(y2)
	
	return coordinates

def findImage(label_name,image_list):
	'''
	removes corresponding images to excluded labels
	'''
	
	for image in image_list:
		if label_name.split('.')[0] == image.split('.')[0]:
			return image
			#print("Image:{}\t label:{}".format(image,label))
			#counter += 1		
	#print("Added {} into dictionary".format(counter))				
	
	
#recall()

#ground_truth()
