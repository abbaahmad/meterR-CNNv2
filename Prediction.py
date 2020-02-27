import os
import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms
import torch.nn as nn


def predict(image,model_path,threshold=0.8):

	classes = ('display','background')

	img_array = Image.open(image).convert('RGB')
	preprocess = transforms.Compose([
					transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485,0.456,0.406],
						std=[0.229,0.224,0.225])
	])
	img = preprocess(img_array)
	image = img.reshape(-1,3,224,224)

	checkpoint = torch.load(model_path)	
	model = models.alexnet(pretrained=False)
	model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features,2)			
	model.load_state_dict(checkpoint['model_state_dict']) 

	#print(model)

	model.eval()

	output = model(image)

	_,preds_tensor= torch.max(output,1)
	#preds = np.squeeze(preds_tensor.numpy())
	softmax = nn.functional.softmax(output,dim=1)
	argmax = torch.argmax(softmax,dim=1)
	
	if argmax >= threshold:# and classes[argmax] == 'display':
		confidence = classes[argmax]#classes[torch.argmax(softmax,dim=1)]
	else:
		#then there is no display i.e. background
		confidence = classes[1]
	#print("Prediction is {} and softmax is {}".format(preds,softmax))
	#print("Prediction is {} of class {}".format(preds,pred))
	
	#pred_dict = {'pred':pred,'softmax',softmax}
	
	return confidence#, preds_tensor
