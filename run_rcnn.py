import cv2
import argparse
import os
import utils
from Prediction import predict
import selectiveSearch as ss
import pickle
import random
import math

ap = argparse.ArgumentParser()
ap.add_argument("--root",required=True,
				help="path to images/labels")
ap.add_argument("--regions",default=100,
				help="number of regions to use")
				
args=vars(ap.parse_args())


#read images
images = list(sorted(os.listdir(os.path.join(args['root'],'images'))))
#regions = ss.get_regions(imgpath,int(args['regions']))

print("[INFO] Using top-{} Region Proposals".format(args['regions']))

#GroundTruths
image_dict = {}
label_list=[]
label_dir = list(sorted(os.listdir('labels')))
idx = random.randint(0,len(images))

for fl in label_dir:
	fPath = os.path.join('labels',fl)
	with open(fPath,'r') as f:
		label = f.read().split()
		
	label = [x for x in label]
	if len(label) > 5:
		print("{} contains more than 5 entries".format(fl))
	#print("Values in file are: {}".format(label))
	label_list.append(label)
	corresponding_image = utils.findImage(fl,images)
	if corresponding_image != None:
		image_dict[corresponding_image]=label
	


counter = 0
pred =[]
good_regions=[]
image = images[idx]
label = image_dict[image]
#for image,label in image_dict.items():
print("[INFO] checking {}-th file".format(idx+1))
img_path = os.path.join(args['root'],'images',image)
img = cv2.imread(img_path)
Rects = ss.get_regions(img_path,args['regions'])
coord = utils.calculate_dims(label[1:])
coord = [int(math.ceil(x)) for x in coord]
for rect in Rects:
	x,y,w,h = rect
	cropped_img = utils.crop_image(img,x,y,x+w,y+w)
	cv2.imwrite('cropped_image.jpg',cropped_img)
	#print("[INFO] saving cropped image")
	confidence = predict('cropped_image.jpg','alexnetv3_model.pth')
	pred.append(confidence)

print("Predictions:\n")
print(pred)
for i in range(len(pred)):
	if pred[i] =='display':
		x,y,w,h = Rects[i]
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1,cv2.LINE_AA)
		cv2.rectangle(img,(coord[0],coord[1]),(coord[2],coord[3]),(255,0,255),2,cv2.LINE_AA)
		cv2.imwrite(os.path.join('Predictions',image),img)
	#if counter == 5:
	#	exit()
	#counter = counter +1
	
exit()


#img = cv2.imread(args['image'])

# while counter < int(args['regions']):
	# #Show original
	# #imgOut = img.copy()
	# print("Counter is "+str(counter))
	# for i,rect in enumerate(regions):
			# #x,y,w,h = rect
			
			# #cropped_img = utils.crop_image(img,x,y,x+w,y+w)
			# #cv2.imwrite('cropped_image.jpg',cropped_img)
			# #print("[INFO] saving cropped image")
			# #print("[INFO] Checking {}/{}".format(i,len(regions)))
			# confidence = predict('cropped_image.jpg','alexnet_model.pth')
			
			# #print("Prediction is {}".format(class_pred))
			# #print("[INFO] path to image")
			# #img_path = os.path.join('images',"img{}.jpeg".format(counter))
			# #print("[INFO] saving file to {}".format(img_path))
			# counter = counter + 1
			# pred_dict[img_path] = confidence
			
			# #print("[INFO] saving image {} of class {}".format(counter-1,class_pred))
		
# with open('Predictions.txt','w') as f:
	# for k,v in pred_dict.items():
		# f.write(str(k)+"---"+str(v)+"\n")
